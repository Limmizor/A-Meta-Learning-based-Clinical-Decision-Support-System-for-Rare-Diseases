"""热力图修复效果对比：左=原版(eigen_smooth+layer4)  右=新版(多层聚合+肺野掩膜)"""
import os
import sys
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pf_diagnosis_service import PFDianosisService, PRESENTATION_CONFIG

MODEL_PATH = os.path.join('models', 'best_maml_fold1.pth')
OUT_DIR = os.path.join('static', 'gradcam', '_test_compare')
os.makedirs(OUT_DIR, exist_ok=True)

# 选几个 OSIC 患者、每人取中段 4 张切片
OSIC_ROOT = os.path.join('data', 'osic-pulmonary-fibrosis-progression', 'train')
PATIENTS = [
    'ID00007637202177411956430',
    'ID00009637202177434476278',
    'ID00010637202177584971671',
]
SLICE_INDICES = [10, 15, 20]  # 中段切片


def pick_slices(patient_id):
    folder = os.path.join(OSIC_ROOT, patient_id)
    files = sorted([f for f in os.listdir(folder) if f.endswith('.dcm')],
                   key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(folder, files[i]) for i in SLICE_INDICES if i < len(files)]


def build_old_overlay(service, image_path, image_tensor, predicted_class):
    """复现修改前的实现：layer4 + eigen_smooth，无肺野掩膜，无分位阈值"""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    old_cam = GradCAM(model=service.model, target_layers=[service.model.layer4])
    cam = old_cam(
        input_tensor=image_tensor,
        targets=[ClassifierOutputTarget(predicted_class)],
        aug_smooth=False,
        eigen_smooth=True,
    )[0].astype(np.float32)
    base = service._load_display_image(image_path).convert('RGB')
    base_arr = np.asarray(base, dtype=np.float32) / 255.0
    overlay = show_cam_on_image(base_arr, cam, use_rgb=True, image_weight=0.5)
    return Image.fromarray(overlay)


def side_by_side(old_img, new_img, base_img, label):
    """三联图：CT 原图 | 旧版 | 新版"""
    w, h = 224, 224
    pad = 10
    canvas = Image.new('RGB', (w * 3 + pad * 4, h + 40), (255, 255, 255))
    canvas.paste(base_img.resize((w, h)), (pad, 30))
    canvas.paste(old_img.resize((w, h)), (pad * 2 + w, 30))
    canvas.paste(new_img.resize((w, h)), (pad * 3 + w * 2, 30))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for i, txt in enumerate(['CT (lung window)', 'OLD (eigen+layer4)', f'NEW ({label})']):
        draw.text((pad + i * (w + pad) + 4, 8), txt, fill=(0, 0, 0))
    return canvas


def main():
    print('==> 初始化服务...')
    service = PFDianosisService(model_path=MODEL_PATH)
    service._adapt()

    results = []
    for pid in PATIENTS:
        slices = pick_slices(pid)
        if not slices:
            print(f'跳过 {pid}: 无切片')
            continue
        print(f'\n== 患者 {pid}: {len(slices)} 张切片 ==')

        tensors = [service._preprocess_image(p) for p in slices]
        x = torch.cat(tensors, dim=0).to(service.device)
        with torch.no_grad():
            probs = torch.softmax(service.model(x), dim=1).cpu().numpy()
        classes = probs.argmax(axis=1)
        print(f'   逐张类别: {classes.tolist()}, 概率(类1): {probs[:,1].round(3).tolist()}')

        for i, path in enumerate(slices):
            pred_cls = int(classes[i])
            xi = x[i:i + 1]
            base = service._load_display_image(path)

            old_img = build_old_overlay(service, path, xi, pred_cls)
            new_img = service._generate_gradcam_overlay(path, xi, pred_cls)

            label = 'multi-layer+lung-mask'
            triptych = side_by_side(old_img, new_img, base, label)
            out = os.path.join(OUT_DIR, f'{pid}_slice{SLICE_INDICES[i]}_cls{pred_cls}.png')
            triptych.save(out)
            results.append(out)
            print(f'   -> {out}')

    print(f'\n完成，共 {len(results)} 张对比图保存到 {OUT_DIR}')


if __name__ == '__main__':
    main()
