"""
生成 MAML 推理所需的支持集（与 train_maml_final.py 的测试协议一致）：
1. 复现 5 折患者级分层交叉验证（StratifiedKFold, shuffle, random_state=42），取第 1 折；
2. 从第 1 折训练集中按类别各取 2 名患者作为支持患者；
3. 每名支持患者按论文流程取 8 张切片（动态分层采样）；
4. 预处理后保存到 models/ipf_support_set.pt，供 pf_diagnosis_service 推理时做 2 步内循环适应。
"""
import os
import random
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from osic_dataset import IPFDataset, IPFSingleSliceDataset, DEFAULT_CSV_PATH, DEFAULT_DICOM_ROOT


WAYS = 2
SHOTS = 2
SUPPORT_SLICES_PER_PATIENT = 8
NUM_SLICES_PER_PATIENT = 80
FOLD_INDEX = 1          # 使用 best_maml_fold1.pth 对应的第 1 折
VAL_SEED = 12345
TEST_SEED = 67890

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def stratified_sample(indices, num_samples, rng):
    """与 train_maml_final.py 相同的分层采样"""
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


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1) 复现第 1 折的患者级划分（直接由 train.csv 计算全部 86 人，患者顺序与 IPFDataset 一致）
    df = pd.read_csv(DEFAULT_CSV_PATH)
    percent_first = df.groupby('Patient')['Percent'].first()
    low = [pid for pid, p in percent_first.items() if p <= 65]
    high = [pid for pid, p in percent_first.items() if p >= 90]
    patients = low + high
    labels = np.array([1] * len(low) + [0] * len(high))
    print(f"全量患者: 类0 {len(high)} 人, 类1 {len(low)} 人, 共 {len(patients)} 人")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(skf.split(patients, labels))
    train_idx, test_idx = splits[FOLD_INDEX - 1]
    train_pids = [patients[i] for i in train_idx]
    test_pids = [patients[i] for i in test_idx]
    print(f"第 {FOLD_INDEX} 折: 训练 {len(train_pids)} 人, 测试 {len(test_pids)} 人")

    class0 = [p for p, lab in zip(train_pids, [labels[i] for i in train_idx]) if lab == 0]
    class1 = [p for p, lab in zip(train_pids, [labels[i] for i in train_idx]) if lab == 1]
    print(f"训练集: 类0 {len(class0)} 人, 类1 {len(class1)} 人")

    # 2) 每类随机取 2 名支持患者（固定种子，保证可复现）
    rng = random.Random(TEST_SEED + FOLD_INDEX)
    support_pids = rng.sample(class0, SHOTS) + rng.sample(class1, SHOTS)
    support_labels = [0] * SHOTS + [1] * SHOTS
    print("支持患者:", list(zip(support_pids, support_labels)))

    # 3) 预处理支持患者切片
    ipf = IPFDataset(csv_path=DEFAULT_CSV_PATH, dicom_root=DEFAULT_DICOM_ROOT,
                     num_slices=NUM_SLICES_PER_PATIENT, transform=BASE_TRANSFORM,
                     patient_ids=support_pids)
    single = IPFSingleSliceDataset(ipf)

    support_x, support_y = [], []
    slice_rng = random.Random(VAL_SEED + FOLD_INDEX)
    for pid, lab in zip(support_pids, support_labels):
        indices = single.patient_to_indices[pid]
        chosen = stratified_sample(indices, SUPPORT_SLICES_PER_PATIENT, slice_rng)
        for idx in chosen:
            img, _, _ = single[idx]
            support_x.append(img)
            support_y.append(lab)

    support_x = torch.stack(support_x).float()
    support_y = torch.tensor(support_y, dtype=torch.long)
    print(f"支持集形状: x {tuple(support_x.shape)}, y {tuple(support_y.shape)}, "
          f"分布: {dict(zip(*torch.unique(support_y, return_counts=True)))}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'models', 'ipf_support_set.pt')
    torch.save({'support_x': support_x, 'support_y': support_y,
                'support_pids': support_pids, 'support_labels': support_labels,
                'fold': FOLD_INDEX}, out_path)
    print(f"已保存: {out_path}")


if __name__ == '__main__':
    main()
