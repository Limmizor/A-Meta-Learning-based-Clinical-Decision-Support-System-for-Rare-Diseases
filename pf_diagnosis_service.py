import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import numpy as np
from database import Database

# 假设你的模型输出6类（对应6种肺纤维化亚型）
NUM_CLASSES = 6

class PFDianosisService:   # 注意：类名中的拼写错误“Dianosis”可保留，以免影响已有引用
    def __init__(self, model_path='./models/pf_maml_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # 疾病名称映射（必须与数据库中的name字段完全一致）
        self.disease_map = {
            1: '特发性肺纤维化(IPF)',
            2: '非特异性间质性肺炎(NSIP)',
            3: '结缔组织病相关肺纤维化(CTD-IP)',
            4: '过敏性肺炎(HP)',
            5: '石棉肺',
            6: '药物性肺纤维化'
        }

    def _load_model(self, path):
        """加载预训练模型，若文件不存在则使用随机初始化模型（仅测试用）"""
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"成功加载肺纤维化模型: {path}")
        else:
            print(f"警告: 模型文件 {path} 不存在，使用随机初始化模型（仅测试用）")
        model = model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_path):
        """预处理单张CT切片，返回模型输入tensor"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

    def predict_from_paths(self, image_paths, patient_id):
        """
        直接对图像路径列表进行预测，返回详细诊断信息
        返回值：
            predictions: list of dict [{'disease_name': str, 'confidence': float}]
            heatmap_url: str (URL路径)
            lesion_area_ratio: float (0~1)
            distribution_range: str
            imaging_findings: str
            suggestions: str
        """
        # ========== 真实模型推理部分（TODO：替换为你的模型逻辑） ==========
        # 当前为模拟数据，便于前端调试
        predictions = [
            {'disease_name': '特发性肺纤维化(IPF)', 'confidence': 0.85},
            {'disease_name': '非特异性间质性肺炎(NSIP)', 'confidence': 0.10},
            {'disease_name': '结缔组织病相关肺纤维化(CTD-IP)', 'confidence': 0.05}
        ]

        # ========== 生成 Grad-CAM 热力图（模拟） ==========
        static_gradcam = os.path.join('static', 'gradcam')
        os.makedirs(static_gradcam, exist_ok=True)
        heatmap_filename = f'heatmap_{patient_id}.png'
        heatmap_path = os.path.join(static_gradcam, heatmap_filename)
        if not os.path.exists(heatmap_path):
            # 创建一个示例红黄色块图（代替真实热力图）
            img = Image.new('RGB', (224, 224), color='darkred')
            draw = ImageDraw.Draw(img)
            draw.rectangle([50, 50, 150, 150], fill='yellow')
            img.save(heatmap_path)
        heatmap_url = f'/static/gradcam/{heatmap_filename}'

        # ========== 量化指标与报告内容（模拟） ==========
        lesion_ratio = 0.32
        distribution = '双肺下叶背段及胸膜下'
        findings = '双肺可见网格影、蜂窝影，以胸膜下为著，伴牵拉性支气管扩张。'
        suggestions = '建议高分辨率CT随访，评估抗纤维化药物治疗。'

        return predictions, heatmap_url, lesion_ratio, distribution, findings, suggestions

    def diagnose_patient(self, patient_id):
        """
        对患者进行肺纤维化诊断，返回格式与原系统兼容：
        [{'disease_id':1, 'disease_name':'...', 'confidence':0.85, 'rank':1}, ...]
        """
        db = Database()
        if not db.connect():
            print("数据库连接失败，返回模拟结果")
            return self._fallback_predictions()

        images = db.get_medical_images(patient_id)
        db.disconnect()

        if not images:
            print(f"患者 {patient_id} 没有上传影像")
            return self._fallback_predictions()

        # 简单取第一张影像（如需多切片融合可后续扩展）
        img_path = os.path.join('static', 'uploads', images[0]['filename'])
        if not os.path.exists(img_path):
            return self._fallback_predictions()

        input_tensor = self._preprocess_image(img_path).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        predictions = []
        for disease_id, disease_name in self.disease_map.items():
            predictions.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'confidence': float(probs[disease_id-1]),
                'rank': 0
            })
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        for i, p in enumerate(predictions):
            p['rank'] = i + 1

        return predictions[:3]  # 返回前3个最可能的亚型

    def _fallback_predictions(self):
        """当模型或数据出错时返回模拟预测结果"""
        return [
            {'disease_id': 1, 'disease_name': '特发性肺纤维化(IPF)', 'confidence': 0.80, 'rank': 1},
            {'disease_id': 2, 'disease_name': '非特异性间质性肺炎(NSIP)', 'confidence': 0.12, 'rank': 2},
            {'disease_id': 3, 'disease_name': '结缔组织病相关肺纤维化(CTD-IP)', 'confidence': 0.08, 'rank': 3}
        ]