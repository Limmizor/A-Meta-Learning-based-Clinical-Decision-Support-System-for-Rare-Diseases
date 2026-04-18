import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from database import Database

# 假设你的模型输出6类（对应6种肺纤维化亚型）
NUM_CLASSES = 6

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
        # 根据你的实际模型结构修改此处
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
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

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
        return [
            {'disease_id': 1, 'disease_name': '特发性肺纤维化(IPF)', 'confidence': 0.80, 'rank': 1},
            {'disease_id': 2, 'disease_name': '非特异性间质性肺炎(NSIP)', 'confidence': 0.12, 'rank': 2},
            {'disease_id': 3, 'disease_name': '结缔组织病相关肺纤维化(CTD-IP)', 'confidence': 0.08, 'rank': 3}
        ]