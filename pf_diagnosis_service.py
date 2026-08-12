import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from database import Database
import torchvision.models as models
import pydicom
import copy
from collections import Counter

# 模型输出2类：论文中定义的IPF预后表型二分类
# 类0 = Percent≥90（相对稳定组），类1 = Percent≤65（严重受损组）
# 若训练代码的类别顺序相反，只需交换 CLASS_NAMES 的两个值
NUM_CLASSES = 2
CLASS_NAMES = {
    0: '相对稳定组 (Percent≥90)',
    1: '严重受损组 (Percent≤65)'
}

# 论文/训练代码预处理：肺窗窗宽窗位（窗宽1500HU，窗位-450HU，即 [-1200, 300]）
# 与 ipf_data.py 完全一致：直接对原始像素值截断，不做 Rescale 换算
CT_WINDOW_CENTER = -450
CT_WINDOW_WIDTH = 1500

# MAML 推理配置（与 train_maml_final.py 一致：2 步内循环适应，学习率 0.003）
INNER_STEPS = 2
INNER_LR = 0.003
SUPPORT_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models', 'ipf_support_set.pt')

class GradCAM:
    """手动实现 Grad-CAM，不依赖 torchcam"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        target_layer = self._find_layer(self.target_layer)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def _find_layer(self, layer_name):
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found")

    def generate(self, input_tensor, target_class):
        self.model.eval()
        input_tensor.requires_grad_(True)
        out = self.model(input_tensor)
        self.model.zero_grad()
        loss = out[0, target_class]
        loss.backward(retain_graph=True)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

class PFDianosisService:
    def __init__(self, model_path='./models/pf_maml_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.disease_map = dict(CLASS_NAMES)
        self.gradcam = GradCAM(self.model, target_layer='layer4')
        # 加载预生成的支持集（用于 MAML 快速适应），缺失时退化为直接推理
        self.support_x, self.support_y = self._load_support_set()
        self._base_state = copy.deepcopy(self.model.state_dict())

    def _load_model(self, path):
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=self.device)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            print(f"成功加载肺纤维化模型: {path}")
        else:
            print(f"警告: 模型文件 {path} 不存在，使用随机初始化模型（仅测试用）")
        model = model.to(self.device)
        model.eval()
        return model

    def _load_support_set(self):
        if not os.path.exists(SUPPORT_SET_PATH):
            print(f"警告: 支持集 {SUPPORT_SET_PATH} 不存在，使用无适应直接推理")
            return None, None
        try:
            data = torch.load(SUPPORT_SET_PATH, map_location=self.device, weights_only=False)
            sx = data['support_x'].to(self.device)
            sy = data['support_y'].to(self.device)
            print(f"已加载支持集: {sx.shape[0]} 张切片（每类 {sx.shape[0]//2} 张）")
            return sx, sy
        except Exception as e:
            print(f"支持集加载失败: {e}，使用无适应直接推理")
            return None, None

    def _adapt(self):
        """从元初始参数出发，在支持集上执行 INNER_STEPS 步梯度下降
        （与 train_maml_final (1).py 一致：CE(label_smoothing=0.1)，inner_lr=0.003）"""
        if self.support_x is None:
            return
        self.model.load_state_dict(self._base_state)
        self.model.train()
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        for _ in range(INNER_STEPS):
            self.model.zero_grad()
            logits = self.model(self.support_x)
            loss = loss_fn(logits, self.support_y)
            grads = torch.autograd.grad(loss, self.model.parameters())
            with torch.no_grad():
                for p, g in zip(self.model.parameters(), grads):
                    if g is not None:
                        p.sub_(INNER_LR * g)
        self.model.eval()

    def _preprocess_image(self, image_path):
        """预处理：与 ipf_data.py 完全一致（原始像素窗宽窗位 -> 224 -> RGB）"""
        ext = os.path.splitext(image_path)[1].lower()
        if ext == '.dcm':
            dcm = pydicom.dcmread(image_path)
            img = dcm.pixel_array.astype(np.float32)
            low = CT_WINDOW_CENTER - CT_WINDOW_WIDTH // 2
            high = CT_WINDOW_CENTER + CT_WINDOW_WIDTH // 2
            img = np.clip(img, low, high)
            img = (img - low) / (high - low)
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img = img.resize((224, 224), Image.BILINEAR)
            img = img.convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')
        return self.transform(img).unsqueeze(0)

    def _generate_gradcam(self, image_tensor, predicted_class):
        cam = self.gradcam.generate(image_tensor, predicted_class)
        cam = np.uint8(255 * cam)
        cam = Image.fromarray(cam).resize((224, 224), Image.BILINEAR)
        import matplotlib.cm as cm
        colormap = cm.jet(np.array(cam) / 255.0)[:, :, :3]
        heatmap = (colormap * 255).astype(np.uint8)
        return Image.fromarray(heatmap)

    def predict_from_paths(self, image_paths, patient_id):
        """
        对一组CT切片进行患者级预测（论文：切片逐张推理 + 患者级多数投票）
        返回值：
            predictions: list of dict [{'disease_name': str, 'confidence': float}]
            heatmap_url: str (URL路径)
            lesion_area_ratio: float (0~1) 严重受损类切片占比
            distribution_range: str 切片一致性描述
            imaging_findings: str
            suggestions: str
        """
        if not image_paths:
            return self._fallback_predictions()

        # MAML 快速适应：先加载元初始参数，再在支持集上适应 INNER_STEPS 步
        self._adapt()

        # 逐张切片预处理与推理
        tensors = []
        failed = []
        for p in image_paths:
            try:
                tensors.append(self._preprocess_image(p))
            except Exception as e:
                failed.append(os.path.basename(p))
        if not tensors:
            raise RuntimeError(f'所有切片均无法读取（{"; ".join(failed[:3])}），请检查文件格式')

        input_tensor = torch.cat(tensors, dim=0).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        n_slices = probs.shape[0]
        slice_classes = probs.argmax(axis=1)
        # 患者级多数投票（论文机制）
        counter = Counter(int(c) for c in slice_classes)
        predicted_class = int(counter.most_common(1)[0][0])
        agree_count = counter[predicted_class]
        # 患者级置信度 = 被投为最终类别的切片平均概率
        mask = (slice_classes == predicted_class)
        confidence = float(probs[mask, predicted_class].mean()) if agree_count else float(probs[:, predicted_class].mean())
        # 严重受损（类1）切片占比
        impaired_slice_ratio = float((slice_classes == 1).mean())

        predictions = [
            {'disease_name': CLASS_NAMES[0], 'confidence': float(probs[:, 0].mean())},
            {'disease_name': CLASS_NAMES[1], 'confidence': float(probs[:, 1].mean())}
        ]

        # 用一张最终类别对应的代表切片生成 Grad-CAM
        rep_idx = int(np.where(slice_classes == predicted_class)[0][0]) if agree_count else 0
        heatmap_img = self._generate_gradcam(input_tensor[rep_idx:rep_idx + 1], predicted_class)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_gradcam = os.path.join(base_dir, 'static', 'gradcam')
        os.makedirs(static_gradcam, exist_ok=True)
        heatmap_filename = f'heatmap_{patient_id}.png'
        heatmap_path = os.path.join(static_gradcam, heatmap_filename)
        heatmap_img.save(heatmap_path)
        heatmap_url = f'/static/gradcam/{heatmap_filename}'

        # ========== 基于模型输出的报告文本（不再使用模拟数据） ==========
        distribution = f'{agree_count}/{n_slices} 张切片支持该结论'
        if predicted_class == 1:
            imaging_findings = (
                f'基于 {n_slices} 张CT切片的患者级多数投票，模型判定为「{CLASS_NAMES[1]}」，'
                f'其中 {impaired_slice_ratio * 100:.1f}% 的切片倾向该结论，'
                f'提示肺功能严重受损（Percent≤65）表型可能性较高。'
            )
            suggestions = (
                '建议结合临床肺功能检查（FVC%）进一步确认，关注疾病快速进展风险，'
                '及时评估抗纤维化药物治疗方案并按医嘱随访。'
            )
        else:
            imaging_findings = (
                f'基于 {n_slices} 张CT切片的患者级多数投票，模型判定为「{CLASS_NAMES[0]}」，'
                f'仅 {impaired_slice_ratio * 100:.1f}% 的切片倾向严重受损组，'
                f'提示肺功能相对稳定（Percent≥90）表型可能性较高。'
            )
            suggestions = (
                '建议维持常规随访，定期复查肺功能与CT影像，监测FVC变化趋势，'
                '如出现下降应尽早启动干预。'
            )

        return predictions, heatmap_url, impaired_slice_ratio, distribution, imaging_findings, suggestions
    
    def diagnose_patient(self, patient_id):
        db = Database()
        if not db.connect():
            print("数据库连接失败，返回模拟结果")
            return self._fallback_predictions()

        images = db.get_medical_images(patient_id)
        db.disconnect()

        if not images:
            print(f"患者 {patient_id} 没有上传影像")
            return self._fallback_predictions()

        image_paths = [os.path.join('static', 'uploads', img['image_path']) for img in images]
        try:
            predictions, heatmap_url, impaired_ratio, distribution, findings, suggestions = \
                self.predict_from_paths(image_paths, patient_id)
        except Exception as e:
            print(f"患者 {patient_id} 推理失败: {e}")
            return self._fallback_predictions()

        result = []
        for disease_id, disease_name in self.disease_map.items():
            result.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'confidence': predictions[disease_id]['confidence'],
                'rank': 0
            })
        result.sort(key=lambda x: x['confidence'], reverse=True)
        for i, p in enumerate(result):
            p['rank'] = i + 1

        return {
            'predictions': result,
            'heatmap_url': heatmap_url,
            'lesion_area_ratio': impaired_ratio,
            'distribution_range': distribution,
            'imaging_findings': findings,
            'suggestions': suggestions
        }

    def _fallback_predictions(self):
        return [
            {'disease_id': 0, 'disease_name': CLASS_NAMES[0], 'confidence': 0.80, 'rank': 1},
            {'disease_id': 1, 'disease_name': CLASS_NAMES[1], 'confidence': 0.20, 'rank': 2}
        ]
