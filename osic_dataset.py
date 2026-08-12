import os
import random
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
import torch
from torch.utils.data import Dataset

# 项目内默认数据路径（完整 OSIC 数据集解压在 data/osic-pulmonary-fibrosis-progression/）
_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data', 'osic-pulmonary-fibrosis-progression')
DEFAULT_CSV_PATH = os.path.join(_BASE, 'train.csv')
DEFAULT_DICOM_ROOT = os.path.join(_BASE, 'train')


class IPFDataset(Dataset):
    def __init__(self, csv_path=DEFAULT_CSV_PATH, dicom_root=DEFAULT_DICOM_ROOT,
                 num_slices=30, transform=None, patient_ids=None):
        """
        标签定义：Percent ≤ 65 -> 1（严重受损组），Percent ≥ 90 -> 0（相对稳定组）
        剔除中间患者 (65 < Percent < 90)
        """
        self.df = pd.read_csv(csv_path)
        self.dicom_root = dicom_root
        self.num_slices = num_slices
        self.transform = transform

        percent_first = self.df.groupby('Patient')['Percent'].first()
        low_patients = [pid for pid, p in percent_first.items() if p <= 65]
        high_patients = [pid for pid, p in percent_first.items() if p >= 90]
        selected_patients = low_patients + high_patients
        labels = []
        for pid in selected_patients:
            if pid in low_patients:
                labels.append(1)
            else:
                labels.append(0)
        self.patients = selected_patients
        self.labels = np.array(labels)
        print(f"标签分布: Percent ≤ 65: {len(low_patients)} 人, Percent ≥ 90: {len(high_patients)} 人")
        print(f"总计: {len(self.patients)} 人")

        if patient_ids is not None:
            mask = [pid in patient_ids for pid in self.patients]
            self.patients = [pid for pid, keep in zip(self.patients, mask) if keep]
            self.labels = self.labels[mask]

        valid_patients = []
        valid_labels = []
        for pid, lab in zip(self.patients, self.labels):
            patient_folder = os.path.join(self.dicom_root, str(pid))
            if os.path.isdir(patient_folder):
                dcm_files = [f for f in os.listdir(patient_folder) if f.endswith('.dcm')]
                if dcm_files:
                    valid_patients.append(pid)
                    valid_labels.append(lab)
                else:
                    print(f"警告: 患者 {pid} 文件夹存在但无DICOM，已跳过")
            else:
                print(f"警告: 患者 {pid} 文件夹不存在，已跳过")
        self.patients = valid_patients
        self.labels = np.array(valid_labels)
        self.patient_to_idx = {pid: i for i, pid in enumerate(self.patients)}
        print(f"最终有效患者数: {len(self.patients)}")
        print(f"最终分布: 类0 (Percent≥90): {sum(1 for l in self.labels if l==0)}, 类1 (Percent≤65): {sum(1 for l in self.labels if l==1)}")

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient_id = str(self.patients[idx])
        patient_folder = os.path.join(self.dicom_root, patient_id)
        dicom_files = sorted([f for f in os.listdir(patient_folder) if f.endswith('.dcm')])
        if len(dicom_files) == 0:
            raise FileNotFoundError(f"No DICOM files in {patient_folder}")

        total = len(dicom_files)
        if total <= self.num_slices:
            selected_indices = list(range(total))
        else:
            step = total / self.num_slices
            selected_indices = [int(i * step) for i in range(self.num_slices)]

        slices = []
        for idx_selected in selected_indices:
            fname = dicom_files[idx_selected]
            dcm_path = os.path.join(patient_folder, fname)
            try:
                dcm = pydicom.dcmread(dcm_path)
                img = dcm.pixel_array.astype(np.float32)
                center, width = -450, 1500
                low = center - width // 2
                high = center + width // 2
                img = np.clip(img, low, high)
                img = (img - low) / (high - low)
                img = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((224, 224), Image.BILINEAR)
                pil_img = pil_img.convert('RGB')
                if self.transform:
                    pil_img = self.transform(pil_img)
                slices.append(pil_img)
            except Exception as e:
                print(f"警告: 处理患者 {patient_id} 的切片 {fname} 时出错: {e}")
                continue
        while len(slices) < self.num_slices:
            slices.append(slices[-1])
        images = torch.stack(slices)
        return images, self.labels[idx], patient_id


class IPFSingleSliceDataset(Dataset):
    def __init__(self, ipf_dataset):
        self.samples = []
        self.patient_to_indices = {}
        for patient_idx in range(len(ipf_dataset)):
            images, label, patient_id = ipf_dataset[patient_idx]
            for j in range(images.size(0)):
                idx = len(self.samples)
                self.samples.append((images[j], label, patient_id))
                if patient_id not in self.patient_to_indices:
                    self.patient_to_indices[patient_id] = []
                self.patient_to_indices[patient_id].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label, pid = self.samples[idx]
        return img, label, pid


def sample_task_patient_level(dataset, ways, shots, queries,
                              query_slices_per_patient=1,
                              return_counts=False):
    class_to_patients = {}
    for pid, indices in dataset.patient_to_indices.items():
        _, label, _ = dataset[indices[0]]
        label_int = label.item() if torch.is_tensor(label) else label
        class_to_patients.setdefault(label_int, []).append(pid)

    valid_classes = [cls for cls, patients in class_to_patients.items()
                     if len(patients) >= shots + queries]
    if len(valid_classes) < ways:
        raise ValueError(f"Not enough classes: need {ways}, have {len(valid_classes)}")

    selected_classes = random.sample(valid_classes, ways)
    support_x, support_y = [], []
    query_x, query_y = [], []
    query_counts = []

    for cls in selected_classes:
        patients = class_to_patients[cls].copy()
        random.shuffle(patients)
        support_patients = patients[:shots]
        query_patients = patients[shots:shots + queries]

        for pid in support_patients:
            indices = dataset.patient_to_indices[pid]
            chosen_idx = random.choice(indices)
            img, _, _ = dataset[chosen_idx]
            support_x.append(img)
            support_y.append(cls)

        for pid in query_patients:
            indices = dataset.patient_to_indices[pid]
            n_slices = len(indices)
            k = min(query_slices_per_patient, n_slices) if query_slices_per_patient > 0 else n_slices
            if k == 0:
                k = 1
            if k <= n_slices:
                chosen_idxs = random.sample(indices, k)
            else:
                chosen_idxs = random.choices(indices, k=k)
            for idx in chosen_idxs:
                img, _, _ = dataset[idx]
                query_x.append(img)
                query_y.append(cls)
            query_counts.append(k)

    support_x = torch.stack(support_x).float()
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.stack(query_x).float()
    query_y = torch.tensor(query_y, dtype=torch.long)

    if return_counts:
        return support_x, support_y, query_x, query_y, query_counts
    else:
        return support_x, support_y, query_x, query_y


class AllSlicesDataset(Dataset):
    """将 IPFDataset 的所有切片展开为独立样本，用于监督训练"""
    def __init__(self, ipf_dataset):
        self.samples = []
        for patient_idx in range(len(ipf_dataset)):
            images, label, _ = ipf_dataset[patient_idx]
            for j in range(images.size(0)):
                self.samples.append((images[j], label))
        print(f"AllSlicesDataset: {len(self.samples)} 个切片样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return img, label
