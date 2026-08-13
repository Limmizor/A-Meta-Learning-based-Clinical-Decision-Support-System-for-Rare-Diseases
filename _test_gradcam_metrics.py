"""对四个问题各设一个可量化指标，比较旧版 vs 新版热力图。"""
import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pf_diagnosis_service import PFDianosisService
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

MODEL_PATH = os.path.join('models', 'best_maml_fold1.pth')
OSIC_ROOT = os.path.join('data', 'osic-pulmonary-fibrosis-progression', 'train')
PATIENTS = ['ID00007637202177411956430', 'ID00009637202177434476278',
            'ID00010637202177584971671']
SLICE_IDX = [10, 15, 20]


def cam_old(service, x, cls):
    cam = GradCAM(model=service.model, target_layers=[service.model.layer4])
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)],
               aug_smooth=False, eigen_smooth=True)[0].astype(np.float32)


def cam_new_raw(service, x, cls):
    """新版底层 CAM（未经掩膜/阈值），用于对比原始激活形态"""
    cam = GradCAM(model=service.model,
                  target_layers=[service.model.layer3, service.model.layer4])
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)],
               aug_smooth=False, eigen_smooth=False)[0].astype(np.float32)


def mirror_score(cam):
    """底部镜像伪影度：cam 与其上下翻转的相关度，越高越像"镜像"。"""
    flipped = np.flipud(cam)
    c = cam.flatten() - cam.mean()
    f = flipped.flatten() - flipped.mean()
    denom = (np.linalg.norm(c) * np.linalg.norm(f) + 1e-8)
    return float((c @ f) / denom)


def outside_lung_ratio(cam, lung_mask):
    """肺野外的能量占比：越低越好（不该在胸腔外着色）"""
    cam_pos = np.clip(cam, 0, None)
    total = cam_pos.sum() + 1e-8
    outside = cam_pos[lung_mask < 0.5].sum()
    return float(outside / total)


def high_area_ratio(cam):
    """>0.5 高亮区面积占比：粗略衡量'光晕大小'"""
    cam_n = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return float((cam_n > 0.5).mean())


def mid_band_ratio(cam):
    """过渡带（0.2~0.7）占**整图**比例（绝对面积口径）：越低表示"核心 vs 边界"越清晰"""
    return float(((cam >= 0.2) & (cam <= 0.7)).mean())


def background_noise(cam):
    """归一化后 0.05~0.30 弱激活占**整图**比例（绝对面积口径）：
    反映"肺内背景弱激活噪声"的实际视觉占用，与激活总量脱耦。"""
    return float(((cam >= 0.05) & (cam <= 0.30)).mean())


def main():
    service = PFDianosisService(model_path=MODEL_PATH)
    service._adapt()

    print(f'{"case":38s} | {"hotArea":>8s} {"midBand":>8s} {"bgNoise":>8s}  '
          f'{"hotArea":>8s} {"midBand":>8s} {"bgNoise":>8s}')
    print('-' * 100)
    print(f'{"":38s} | {"--- OLD (eigen+layer4) ---":>27s}   '
          f'{"--- NEW (prior+p80+top-k) ---":>27s}')

    agg = {'old': {'h': [], 'md': [], 'bg': []},
           'new': {'h': [], 'md': [], 'bg': []}}

    for pid in PATIENTS:
        folder = os.path.join(OSIC_ROOT, pid)
        files = sorted([f for f in os.listdir(folder) if f.endswith('.dcm')],
                       key=lambda x: int(os.path.splitext(x)[0]))
        for si in SLICE_IDX:
            if si >= len(files):
                continue
            path = os.path.join(folder, files[si])
            xi = service._preprocess_image(path).to(service.device)
            with torch.no_grad():
                cls = int(torch.softmax(service.model(xi), 1).argmax().item())

            base = service._load_display_image(path)
            gray = np.asarray(base.convert('L'))
            lung = service._lung_mask_from_gray(gray)

            # 旧版（对旧版也做归一化再算 mid/bg 指标，公平比较）
            c_old = cam_old(service, xi, cls)
            c_old_n = (c_old - c_old.min()) / (c_old.max() - c_old.min() + 1e-8)
            h_o = high_area_ratio(c_old)
            md_o = mid_band_ratio(c_old_n)
            bg_o = background_noise(c_old_n)

            # 新版（完整流程：掩膜+纹理先验+p80+gamma+top-k连通域）
            from scipy import ndimage as ndi
            gray_norm = gray.astype(np.float32) / 255.0
            prior = service._lesion_texture_prior(gray_norm, lung)
            c_new = cam_new_raw(service, xi, cls) * lung * (0.4 + 0.6 * prior)
            c_new = ndi.gaussian_filter(c_new, 1.5)
            inside = c_new[lung > 0]
            if inside.size > 0:
                thr = float(np.percentile(inside, 80))
                c_new = np.where(c_new >= thr, c_new - thr, 0.0)
            if c_new.max() > 1e-8:
                c_new = (c_new / c_new.max()) ** 2.0
            c_new = service._keep_top_components(c_new, k=2, min_area_ratio=0.003)
            c_new = c_new * lung
            if c_new.max() > 1e-8:
                c_new = c_new / c_new.max()
            h_n = high_area_ratio(c_new)
            md_n = mid_band_ratio(c_new)
            bg_n = background_noise(c_new)

            tag = f'{pid[-6:]}/slice{si}/cls{cls}'
            print(f'{tag:38s} | {h_o:8.3f} {md_o:8.3f} {bg_o:8.3f}   '
                  f'{h_n:8.3f} {md_n:8.3f} {bg_n:8.3f}')
            agg['old']['h'].append(h_o); agg['old']['md'].append(md_o); agg['old']['bg'].append(bg_o)
            agg['new']['h'].append(h_n); agg['new']['md'].append(md_n); agg['new']['bg'].append(bg_n)

    print('-' * 100)
    print(f'{"MEAN":38s} | '
          f'{np.mean(agg["old"]["h"]):8.3f} {np.mean(agg["old"]["md"]):8.3f} {np.mean(agg["old"]["bg"]):8.3f}   '
          f'{np.mean(agg["new"]["h"]):8.3f} {np.mean(agg["new"]["md"]):8.3f} {np.mean(agg["new"]["bg"]):8.3f}')
    print('\n指标说明:')
    print('  hotArea↓ : 归一化后 >0.5 的高亮面积占比（光晕大小）')
    print('  midBand↓ : 归一化后 0.2~0.7 过渡带占非零像素比例（核心 vs 边界清晰度）')
    print('  bgNoise↓ : 归一化后 0.05~0.30 弱激活占非零像素比例（肺内背景噪声）')


if __name__ == '__main__':
    main()
