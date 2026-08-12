import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import learn2learn as l2l
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score, roc_curve,
    matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from scipy.stats import ttest_1samp
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import copy
import warnings
warnings.filterwarnings('ignore')

from osic_dataset import IPFDataset, IPFSingleSliceDataset, sample_task_patient_level

# ==================== 兼容包装类（已修正类型） ====================
class CompatIPFSingleSliceDataset(IPFSingleSliceDataset):
    """
    为 IPFSingleSliceDataset 补充 patients、get_patient_indices、get_patient_label，
    用于验证集和测试集。
    """
    def __init__(self, ipf_dataset):
        super().__init__(ipf_dataset)
        self.patients = list(self.patient_to_indices.keys())
        self.patient_label = {}
        for pid in self.patients:
            idx = self.patient_to_indices[pid][0]
            _, label, _ = self[idx]
            label_int = label.item() if torch.is_tensor(label) else label
            self.patient_label[pid] = label_int

    def get_patient_indices(self, pid):
        return self.patient_to_indices.get(pid, [])

    def get_patient_label(self, pid):
        # 统一转换为 Python int，避免 numpy.int64 等类型隐患
        return int(self.patient_label.get(pid, -1))

# ==================== 参数配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

ways = 2
shots = 2
queries = 2

support_slices_per_patient = 8
query_slices_per_patient = 8

val_seed = 12345
test_seed = 67890

adapt_steps = 2
inner_lr = 0.003
meta_lr = 0.0008
weight_decay = 1e-4

num_epochs = 100
num_tasks_per_epoch = 120          # 从60增加
num_val_tasks = 300                 # 从100增加
num_test_tasks = 200
val_query_slices_per_patient = 20   # 验证时每个患者query切片数（不用80）
test_query_slices_per_patient = 30  # 测试时每个患者query切片数
label_smoothing = 0.1

n_splits = 5
num_slices_per_patient = 80
VAL_RATIO = 0.25                     # 从0.2提高

csv_path = 'data/osic-pulmonary-fibrosis-progression/train.csv'
dicom_root = 'data/osic-pulmonary-fibrosis-progression/train'

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(42)

# ==================== 数据变换 ====================
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.97, 1.03),
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    ),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== Bootstrap 函数 ====================
def bootstrap_ci_patient_level(y_true, y_pred, n_bootstrap=2000, ci=0.95):
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan
    accs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    if len(accs) == 0:
        return np.nan, np.nan
    lower = np.percentile(accs, (1 - ci) / 2 * 100)
    upper = np.percentile(accs, (1 + ci) / 2 * 100)
    return lower, upper

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=2000, ci=0.95):
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    aucs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if len(aucs) == 0:
        return np.nan, np.nan
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return lower, upper

# ==================== 分层随机采样 ====================
def stratified_sample(indices, num_samples, rng):
    total = len(indices)
    if total <= num_samples:
        return indices.copy()
    step = total / num_samples
    chosen = []
    for i in range(num_samples):
        start = int(i * step)
        end = int((i + 1) * step)
        if start >= total:
            start = total - 1
        if end > total:
            end = total
        if start == end:
            idx = start
        else:
            idx = rng.randint(start, end - 1)
        chosen.append(indices[min(idx, total - 1)])
    return chosen

# ==================== DynamicSliceDataset（已包含防御性检查） ====================
class DynamicSliceDataset:
    def __init__(self, base_dataset, num_slices_per_patient, epoch_seed=42):
        # 防御性检查：如果 base_dataset 是 IPFDataset（返回多张切片），自动转换为 IPFSingleSliceDataset
        if isinstance(base_dataset, IPFDataset):
            base_dataset = IPFSingleSliceDataset(base_dataset)
        self.base = base_dataset
        self.num_slices = num_slices_per_patient
        self.patients = list(self.base.patient_to_indices.keys())
        self.patient_to_indices = self.base.patient_to_indices
        self.epoch_seed = epoch_seed
        self.current_epoch = 0
        self._build_epoch_indices()

    def _build_epoch_indices(self):
        rng = random.Random(self.epoch_seed + self.current_epoch)
        self.epoch_indices = {}
        for pid, indices in self.patient_to_indices.items():
            total = len(indices)
            if total <= self.num_slices:
                self.epoch_indices[pid] = indices.copy()
            else:
                self.epoch_indices[pid] = stratified_sample(indices, self.num_slices, rng)
        self.samples = []
        self.patient_to_epoch_indices = {}
        for pid, idx_list in self.epoch_indices.items():
            for idx in idx_list:
                img, label, _ = self.base[idx]   # 现在 base 必定是 IPFSingleSliceDataset，img 是 [3,224,224]
                sample_idx = len(self.samples)
                self.samples.append((img, label, pid))
                if pid not in self.patient_to_epoch_indices:
                    self.patient_to_epoch_indices[pid] = []
                self.patient_to_epoch_indices[pid].append(sample_idx)

    def rebuild(self, epoch):
        self.current_epoch = epoch
        self._build_epoch_indices()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, pid = self.samples[idx]
        return img, label, pid

    def get_patient_indices(self, pid):
        return self.patient_to_epoch_indices.get(pid, [])

    def get_patient_label(self, pid):
        # 确保返回整数标签（避免张量）
        label = self.base[self.patient_to_indices[pid][0]][1]
        return label.item() if torch.is_tensor(label) else label

# ==================== 多切片任务采样 ====================
def sample_task_multi_slice(dataset, ways, shots, queries,
                            support_slices, query_slices,
                            rng=None, return_counts=False):
    if rng is None:
        rng = random
    class_to_patients = {}
    for pid in dataset.patients:
        indices = dataset.get_patient_indices(pid)
        if not indices:
            continue
        label = dataset.get_patient_label(pid)
        label_int = label.item() if torch.is_tensor(label) else label
        class_to_patients.setdefault(label_int, []).append(pid)
    valid_classes = [cls for cls, patients in class_to_patients.items()
                     if len(patients) >= shots + queries]
    if len(valid_classes) < ways:
        raise ValueError(f"Not enough classes: need {ways}, have {len(valid_classes)}")
    selected_classes = rng.sample(valid_classes, ways)
    support_x, support_y = [], []
    query_x, query_y = [], []
    query_counts = []
    for cls in selected_classes:
        patients = class_to_patients[cls].copy()
        rng.shuffle(patients)
        support_patients = patients[:shots]
        query_patients = patients[shots:shots+queries]
        for pid in support_patients:
            indices = dataset.get_patient_indices(pid)
            chosen_idxs = stratified_sample(indices, support_slices, rng)
            for idx in chosen_idxs:
                img, _, _ = dataset[idx]          # 此时 img 应为 [3,224,224]
                support_x.append(img)
                support_y.append(cls)
        for pid in query_patients:
            indices = dataset.get_patient_indices(pid)
            chosen_idxs = stratified_sample(indices, query_slices, rng)
            for idx in chosen_idxs:
                img, _, _ = dataset[idx]
                query_x.append(img)
                query_y.append(cls)
            query_counts.append(len(chosen_idxs))
    support_x = torch.stack(support_x).float()   # 现在 shape: [N,3,224,224]
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.stack(query_x).float()
    query_y = torch.tensor(query_y, dtype=torch.long)
    if return_counts:
        return support_x, support_y, query_x, query_y, query_counts
    else:
        return support_x, support_y, query_x, query_y

# ==================== 固定验证任务生成 ====================
def generate_fixed_tasks(dataset, ways, shots, queries,
                         support_slices, query_slices, num_tasks, seed):
    rng = random.Random(seed)
    fixed_tasks = []
    for _ in range(num_tasks):
        try:
            sx, sy, qx, qy, q_counts = sample_task_multi_slice(
                dataset, ways, shots, queries,
                support_slices, query_slices,
                rng=rng, return_counts=True
            )
            fixed_tasks.append((sx, sy, qx, qy, q_counts))
        except ValueError:
            continue
    while len(fixed_tasks) < num_tasks:
        try:
            sx, sy, qx, qy, q_counts = sample_task_multi_slice(
                dataset, ways, shots, queries,
                support_slices, query_slices,
                rng=rng, return_counts=True
            )
            fixed_tasks.append((sx, sy, qx, qy, q_counts))
        except ValueError:
            continue
    return fixed_tasks

def evaluate_fixed_tasks(maml_model, fixed_tasks, adapt_steps, loss_fn, device):
    task_accs = []
    all_patient_preds = []
    all_patient_labels = []
    all_patient_probs = []
    for sx, sy, qx, qy, q_counts in fixed_tasks:
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        learner = maml_model.clone()
        for _ in range(adapt_steps):
            preds = learner(sx)
            loss = loss_fn(preds, sy)
            learner.adapt(loss)
        with torch.no_grad():
            logits = learner(qx)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = qy.cpu().numpy()
            probs_np = probs.cpu().numpy()
        start = 0
        patient_correct = 0
        for cnt in q_counts:
            group_preds = preds[start:start+cnt]
            group_labels = labels[start:start+cnt]
            group_probs = probs_np[start:start+cnt, 1]
            majority = Counter(group_preds).most_common(1)[0][0]
            avg_prob = np.mean(group_probs) if len(group_probs) > 0 else 0.5
            if majority == group_labels[0]:
                patient_correct += 1
            start += cnt
            all_patient_preds.append(majority)
            all_patient_labels.append(group_labels[0])
            all_patient_probs.append(avg_prob)
        task_accs.append(patient_correct / len(q_counts))
    return task_accs, np.array(all_patient_preds), np.array(all_patient_labels), np.array(all_patient_probs)

def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, ways)
    return model

# ==================== 患者级评估函数 ====================
def evaluate_patient_level_multi_slice(maml_model, dataset, ways, shots, queries,
                                       support_slices, query_slices_vote,
                                       num_tasks, adapt_steps, loss_fn, device):
    task_accs = []
    all_patient_preds = []
    all_patient_labels = []
    all_patient_probs = []
    rng = random.Random(test_seed)
    for _ in range(num_tasks):
        try:
            sx, sy, qx, qy, q_counts = sample_task_multi_slice(
                dataset, ways, shots, queries,
                support_slices, query_slices_vote,
                rng=rng, return_counts=True
            )
        except ValueError:
            continue
        sx, sy = sx.to(device), sy.to(device)
        qx, qy = qx.to(device), qy.to(device)
        learner = maml_model.clone()
        for _ in range(adapt_steps):
            preds = learner(sx)
            loss = loss_fn(preds, sy)
            learner.adapt(loss)
        with torch.no_grad():
            logits = learner(qx)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = qy.cpu().numpy()
            probs_np = probs.cpu().numpy()
        start = 0
        patient_correct = 0
        for cnt in q_counts:
            group_preds = preds[start:start+cnt]
            group_labels = labels[start:start+cnt]
            group_probs = probs_np[start:start+cnt, 1]
            majority = Counter(group_preds).most_common(1)[0][0]
            avg_prob = np.mean(group_probs) if len(group_probs) > 0 else 0.5
            if majority == group_labels[0]:
                patient_correct += 1
            start += cnt
            all_patient_preds.append(majority)
            all_patient_labels.append(group_labels[0])
            all_patient_probs.append(avg_prob)
        task_accs.append(patient_correct / len(q_counts))
    return task_accs, np.array(all_patient_preds), np.array(all_patient_labels), np.array(all_patient_probs)

# ==================== 单折训练函数 ====================
def train_fold(train_base_ds, val_base_ds, test_base_ds, fold_idx):
    # 训练集用Dynamic（切片动态变化）
    train_ds = DynamicSliceDataset(train_base_ds, num_slices_per_patient, epoch_seed=100 + fold_idx)
    # 验证集和测试集直接使用传入的已包装好的数据集（无需再包装）
    val_ds = val_base_ds   # 已经是 CompatIPFSingleSliceDataset
    test_ds = test_base_ds # 已经是 CompatIPFSingleSliceDataset

    # 生成固定的验证任务（使用 val_query_slices_per_patient = 20）
    val_fixed_tasks = generate_fixed_tasks(
        val_ds, ways, shots, queries,
        support_slices_per_patient,
        val_query_slices_per_patient,
        num_val_tasks, seed=val_seed + fold_idx
    )

    model = create_model().to(DEVICE)
    maml = l2l.algorithms.MAML(model, lr=inner_lr, first_order=False)
    opt = optim.Adam(maml.parameters(), lr=meta_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    val_epochs, val_accs = [], []
    best_val_acc = 0.0
    best_val_auc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        train_ds.rebuild(epoch)
        epoch_losses = []
        for _ in range(num_tasks_per_epoch):
            try:
                sx, sy, qx, qy = sample_task_multi_slice(
                    train_ds, ways, shots, queries,
                    support_slices_per_patient, query_slices_per_patient
                )
            except ValueError:
                continue
            sx, sy = sx.to(DEVICE), sy.to(DEVICE)
            qx, qy = qx.to(DEVICE), qy.to(DEVICE)
            opt.zero_grad()
            learner = maml.clone()
            for _ in range(adapt_steps):
                preds = learner(sx)
                loss = loss_fn(preds, sy)
                learner.adapt(loss)
            loss = loss_fn(learner(qx), qy)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=1.0)
            opt.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            _, val_preds, val_labels, val_probs = evaluate_fixed_tasks(
                maml, val_fixed_tasks, adapt_steps, loss_fn, DEVICE
            )
            val_acc = accuracy_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5
            val_epochs.append(epoch)
            val_accs.append(val_acc)
            print(f"Fold {fold_idx} Epoch {epoch:3d}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val AUC={val_auc:.4f}")

            # 保存标准：AUC优先
            if (val_auc > best_val_auc) or (abs(val_auc - best_val_auc) < 1e-4 and val_acc > best_val_acc):
                best_val_auc = val_auc
                best_val_acc = val_acc
                best_state = copy.deepcopy(maml.state_dict())
                best_epoch = epoch

    if best_state is not None:
        maml.load_state_dict(best_state)

    # 测试时使用 test_query_slices_per_patient = 30
    test_task_accs, test_preds, test_labels, test_probs = evaluate_patient_level_multi_slice(
        maml, test_ds, ways, shots, queries,
        support_slices_per_patient, test_query_slices_per_patient,
        num_test_tasks, adapt_steps, loss_fn, DEVICE
    )
    test_mean = np.mean(test_task_accs)
    test_std = np.std(test_task_accs)

    # 计算患者级指标
    patient_acc = accuracy_score(test_labels, test_preds)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall_score(test_labels, test_preds, average='binary')
    balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds, average='binary')
    precision = precision_score(test_labels, test_preds, average='binary')
    auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) > 1 else 0.5
    mcc = matthews_corrcoef(test_labels, test_preds)
    t_stat, p_val = ttest_1samp(test_task_accs, 0.5)

    acc_ci_low, acc_ci_high = bootstrap_ci_patient_level(test_labels, test_preds)
    auc_ci_low, auc_ci_high = bootstrap_auc_ci(test_labels, test_probs)

    torch.save(maml.state_dict(), f'best_maml_fold{fold_idx}.pth')
    val_curve_df = pd.DataFrame({'epoch': val_epochs, 'val_acc': val_accs})
    val_curve_df.to_csv(f'fold_{fold_idx}_val_curve.csv', index=False)

    if len(np.unique(test_labels)) > 1:
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f'Fold {fold_idx} (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Fold {fold_idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'roc_curve_fold_{fold_idx}.png', dpi=300)
        plt.close()
        has_roc = True
    else:
        print(f"Warning: Fold {fold_idx} has only one class in test set, skipping ROC curve")
        fpr, tpr, has_roc = None, None, False

    return {
        'fold': fold_idx,
        'test_mean': test_mean,
        'test_std': test_std,
        'accuracy': patient_acc,
        'accuracy_ci_low': acc_ci_low,
        'accuracy_ci_high': acc_ci_high,
        'balanced_accuracy': balanced_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'auc_ci_low': auc_ci_low,
        'auc_ci_high': auc_ci_high,
        'mcc': mcc,
        'p_value': p_val,
        't_stat': t_stat,
        'best_val_acc': best_val_acc,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'test_probs': test_probs,
        'fpr': fpr,
        'tpr': tpr,
        'has_roc': has_roc,
        'val_epochs': val_epochs,
        'val_accs': val_accs
    }

# ==================== 主程序 ====================
print("=" * 60)
print("加载数据...")
full_dataset = IPFDataset(csv_path, dicom_root, num_slices=num_slices_per_patient, transform=base_transform)
patients = full_dataset.patients
labels = full_dataset.labels
class0_patients = [p for p, lab in zip(patients, labels) if lab == 0]
class1_patients = [p for p, lab in zip(patients, labels) if lab == 1]
print(f"类别0 (Percent≥90): {len(class0_patients)} 人")
print(f"类别1 (Percent≤65): {len(class1_patients)} 人")
print(f"总计: {len(patients)} 人")

patient_ids = patients
patient_labels = np.array(labels)
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, patient_labels), start=1):
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{n_splits}")
    print(f"{'='*60}")

    train_pids = [patient_ids[i] for i in train_idx]
    test_pids = [patient_ids[i] for i in test_idx]

    train_c0 = [p for p in train_pids if p in class0_patients]
    train_c1 = [p for p in train_pids if p in class1_patients]

    random.seed(42 + fold)
    # 使用 VAL_RATIO=0.25 划分验证集，确保每类至少3人
    val_c0 = random.sample(train_c0, max(3, int(len(train_c0) * VAL_RATIO)))
    val_c1 = random.sample(train_c1, max(3, int(len(train_c1) * VAL_RATIO)))
    val_pids = val_c0 + val_c1
    new_train_pids = [p for p in train_pids if p not in val_pids]

    print(f"训练: {len(new_train_pids)} 人")
    print(f"验证: {len(val_pids)} 人 (类0:{len(val_c0)}, 类1:{len(val_c1)})")
    print(f"测试: {len(test_pids)} 人")

    # 创建数据集（全部使用关键字参数，避免位置参数混乱）
    train_ipf = IPFDataset(
        csv_path=csv_path,
        dicom_root=dicom_root,
        num_slices=num_slices_per_patient,
        transform=train_transform,
        patient_ids=new_train_pids
    )
    val_ipf = IPFDataset(
        csv_path=csv_path,
        dicom_root=dicom_root,
        num_slices=num_slices_per_patient,
        transform=base_transform,
        patient_ids=val_pids
    )
    test_ipf = IPFDataset(
        csv_path=csv_path,
        dicom_root=dicom_root,
        num_slices=num_slices_per_patient,
        transform=base_transform,
        patient_ids=test_pids
    )

    train_base_ds = IPFSingleSliceDataset(train_ipf)
    # 验证集和测试集使用 CompatIPFSingleSliceDataset 包装（仅一次）
    val_base_ds   = CompatIPFSingleSliceDataset(val_ipf)
    test_base_ds  = CompatIPFSingleSliceDataset(test_ipf)

    res = train_fold(train_base_ds, val_base_ds, test_base_ds, fold)
    fold_results.append(res)

# ==================== 汇总结果 ====================
print("\n" + "=" * 60)
print("5折交叉验证结果汇总")
print("=" * 60)

print("\n各折详细结果:")
print(f"{'Fold':>6} {'Acc':>8} {'BalAcc':>8} {'Sen':>8} {'Spe':>8} {'AUC':>8} {'MCC':>8} {'BestEpoch':>10}")
print("-" * 75)
for r in fold_results:
    print(f"{r['fold']:>6} {r['accuracy']:>8.4f} {r['balanced_accuracy']:>8.4f} "
          f"{r['sensitivity']:>8.4f} {r['specificity']:>8.4f} {r['auc']:>8.4f} "
          f"{r['mcc']:>8.4f} {r['best_epoch']:>10d}")

print("\n各折最佳验证性能:")
for r in fold_results:
    print(f"  Fold {r['fold']}: Best Epoch={r['best_epoch']}, Best Val Acc={r['best_val_acc']:.4f}, Best Val AUC={r['best_val_auc']:.4f}")

# ==================== 整体Bootstrap CI ====================
print("\n" + "=" * 60)
print("整体性能（5折汇总后Bootstrap）")
print("=" * 60)

all_test_labels = np.concatenate([r['test_labels'] for r in fold_results])
all_test_preds = np.concatenate([r['test_preds'] for r in fold_results])
all_test_probs = np.concatenate([r['test_probs'] for r in fold_results])

overall_acc = accuracy_score(all_test_labels, all_test_preds)
overall_auc = roc_auc_score(all_test_labels, all_test_probs) if len(np.unique(all_test_labels)) > 1 else 0.5
overall_bal_acc = balanced_accuracy_score(all_test_labels, all_test_preds)
overall_f1 = f1_score(all_test_labels, all_test_preds, average='binary')
overall_mcc = matthews_corrcoef(all_test_labels, all_test_preds)

overall_acc_ci_low, overall_acc_ci_high = bootstrap_ci_patient_level(
    all_test_labels, all_test_preds
)
overall_auc_ci_low, overall_auc_ci_high = bootstrap_auc_ci(
    all_test_labels, all_test_probs
)

print(f"Overall Accuracy: {overall_acc:.4f}  (95% CI: {overall_acc_ci_low:.4f} - {overall_acc_ci_high:.4f})")
print(f"Overall Balanced Accuracy: {overall_bal_acc:.4f}")
print(f"Overall AUC: {overall_auc:.4f}  (95% CI: {overall_auc_ci_low:.4f} - {overall_auc_ci_high:.4f})")
print(f"Overall F1: {overall_f1:.4f}")
print(f"Overall MCC: {overall_mcc:.4f}")

df_folds = pd.DataFrame(fold_results)
df_folds.to_csv('cv_fold_results.csv', index=False)
print("\n各折结果已保存到 cv_fold_results.csv")

# ==================== ROC曲线 ====================
valid_folds = [r for r in fold_results if r['has_roc']]
if valid_folds:
    plt.figure(figsize=(8, 6))
    for r in valid_folds:
        plt.plot(r['fpr'], r['tpr'], alpha=0.3, color='blue', linewidth=1)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('5-Fold ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_curves_overlay.png', dpi=300)
    plt.close()
    print("ROC曲线已保存到 roc_curves_overlay.png")
else:
    print("警告: 无可用的ROC曲线")

print("\n所有任务完成！")
