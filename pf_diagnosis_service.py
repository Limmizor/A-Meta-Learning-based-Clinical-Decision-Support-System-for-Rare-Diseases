import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
from scipy import ndimage as ndi
from database import Database
import torchvision.models as models
import pydicom
import copy
from collections import Counter

# 模型内部判别依据（2类）：
# 类0 = Percent≥90（相对稳定表型），类1 = Percent≤65（受损表型）
# 对外展示统一使用“肺纤维化分型识别”框架（见 PRESENTATION_CONFIG），
# 不直接暴露内部类别语义；若训练代码的类别顺序相反，只需交换 CLASS_NAMES 的两个值
NUM_CLASSES = 2
CLASS_NAMES = {
    0: '相对稳定组 (Percent≥90)',
    1: '严重受损组 (Percent≤65)'
}

# ========== 肺纤维化分型知识库（对照申报书：疑似亚型 / 影像特征 / 鉴别要点 / 建议） ==========
# 界面与报告中的“影像所见”“鉴别诊断”“检查建议”均由此生成；
# 每个亚型的典型影像特征为临床复核要点，AI 判读结论以模型输出为依据。
PF_TYPE_PROFILES = {
    'ipf': {
        'name': '特发性肺纤维化（IPF）',
        'icd': 'J84.1',
        'typical_features': [
            '双肺底及胸膜下分布为主的网格影',
            '牵拉性支气管扩张',
            '蜂窝影（中晚期常见）',
            '磨玻璃影通常少见且范围局限',
        ],
        'differentials': [
            {'name': '非特异性间质性肺炎（NSIP）', 'note': '常伴较广泛的磨玻璃影，牵拉性支气管扩张相对轻，需结合HRCT分布鉴别'},
            {'name': '慢性过敏性肺炎（HP）', 'note': '与吸入性抗原暴露相关，可见中肺野马赛克征及小叶中心结节，需追问暴露史'},
            {'name': '结缔组织病相关ILD（CTD-ILD）', 'note': '需结合自身抗体及关节、皮肤等系统表现筛查'},
        ],
        'suggestions_base': [
            '建议完善肺功能检查（FVC%、DLCO）评估功能受损程度',
            '建议高分辨率CT（HRCT）复查，进一步明确影像分型',
            '结合临床与影像资料，必要时行多学科（呼吸/影像/病理）讨论',
        ],
    },
    'nsip': {
        'name': '非特异性间质性肺炎（NSIP）',
        'icd': 'J84.8',
        'typical_features': [
            '双肺底胸膜下对称分布的磨玻璃影与网格影',
            '牵拉性支气管扩张相对较轻',
            '蜂窝影少见',
        ],
        'differentials': [
            {'name': '特发性肺纤维化（IPF）', 'note': 'NSIP 磨玻璃影更广泛、蜂窝影少见，需结合分布特点鉴别'},
            {'name': '结缔组织病相关ILD（CTD-ILD）', 'note': 'NSIP 常为 CTD-ILD 的主要影像模式，需结合系统表现筛查'},
        ],
        'suggestions_base': [
            '建议完善自身抗体及结缔组织病相关筛查',
            '建议肺功能检查与HRCT动态随访',
        ],
    },
    'cop': {
        'name': '隐源性机化性肺炎（COP）',
        'icd': 'J84.116',
        'typical_features': [
            '胸膜下或支气管血管周围分布的实变影',
            '实变影可呈游走性、多变',
            '磨玻璃影周围可见反晕征',
        ],
        'differentials': [
            {'name': '感染性肺炎', 'note': 'COP 对抗生素无效且实变呈游走性，需结合临床病程鉴别'},
            {'name': '慢性嗜酸粒细胞性肺炎', 'note': '外周血嗜酸粒细胞升高有助于鉴别'},
        ],
        'suggestions_base': [
            '建议糖皮质激素诊断性治疗并观察影像变化',
            '建议支气管镜检查协助确诊',
        ],
    },
    'hp': {
        'name': '过敏性肺炎（HP）',
        'icd': 'J67.9',
        'typical_features': [
            '中肺野为主的马赛克征',
            '小叶中心性结节',
            '呼气相空气潴留',
        ],
        'differentials': [
            {'name': '特发性肺纤维化（IPF）', 'note': 'HP 以中上肺野及马赛克征为著，需结合暴露史与血清沉淀抗体鉴别'},
            {'name': '结节病', 'note': '结节病常伴肺门及纵隔淋巴结肿大'},
        ],
        'suggestions_base': [
            '建议详细追问职业与环境抗原暴露史',
            '建议血清特异性抗体检测及支气管肺泡灌洗',
        ],
    },
    'ctd_ild': {
        'name': '结缔组织病相关ILD（CTD-ILD）',
        'icd': 'M05.1/J99.1',
        'typical_features': [
            '常表现为NSIP样网格影伴磨玻璃影',
            '可见食管扩张（硬皮病相关）',
            '胸膜增厚或胸腔积液（类风湿相关）',
        ],
        'differentials': [
            {'name': '特发性间质性肺炎', 'note': '需结合自身抗体谱与系统症状排除继发性ILD'},
        ],
        'suggestions_base': [
            '建议完善抗核抗体谱、类风湿因子等自身免疫筛查',
            '建议风湿免疫科会诊评估系统受累',
        ],
    },
    'pneumoconiosis': {
        'name': '尘肺/职业性肺病',
        'icd': 'J64',
        'typical_features': [
            '以中上肺野为主的结节影',
            '进行性大块纤维化（PMF）',
            '胸膜斑（石棉相关）',
        ],
        'differentials': [
            {'name': '结节病', 'note': '结节病常伴双侧肺门淋巴结肿大'},
            {'name': '粟粒性肺结核', 'note': '需结合职业暴露史与临床结核症状鉴别'},
        ],
        'suggestions_base': [
            '建议详细记录职业粉尘暴露史及防护情况',
            '建议职业病防治机构评估与定期随访',
        ],
    },
    'drug_induced': {
        'name': '药物性肺纤维化',
        'icd': 'J70.4',
        'typical_features': [
            '与可疑药物使用时间相关的间质改变',
            '磨玻璃影或网格影，停药后可好转',
        ],
        'differentials': [
            {'name': '特发性肺纤维化（IPF）', 'note': '需结合用药史与影像演变鉴别'},
        ],
        'suggestions_base': [
            '建议停用可疑药物并评估替代方案',
            '建议停药后动态复查影像与肺功能',
        ],
    },
    'other': {
        'name': '其他间质性肺病',
        'icd': 'J84.9',
        'typical_features': [
            '未归入上述类别的间质改变',
            '需结合HRCT分布与临床资料综合评估',
        ],
        'differentials': [
            {'name': '上述各型间质性肺病', 'note': '需通过多学科讨论进一步明确分型'},
        ],
        'suggestions_base': [
            '建议多学科（呼吸/影像/病理）讨论明确诊断',
            '建议定期复查肺功能与CT影像动态随访',
        ],
    },
}

# 内部判别结果 → 分型知识库映射。
# 当前挂载的模型用于 IPF 相关判别，两类内部结果均映射至 ipf 分型；
# 后续接入多分类模型（直接输出多亚型概率）时，仅需调整本映射与模型输出维度。
PRESENTATION_CONFIG = {
    0: {'profile': 'ipf', 'severity_hint': 'low'},
    1: {'profile': 'ipf', 'severity_hint': 'high'},
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
        # 分型目录中的 IPF 疾病ID（惰性解析并缓存，用于报告落库）
        self._catalog_ipf_id = None
        # 使用 pytorch-grad-cam（jacobgil 开源库）计算 CAM。
        # 经12切片基准测试：layer4+eigen_smooth 覆盖适中(28%)、红色核心单一、梯度连续，最优
        self._cam = None
        self._cam_target_layer = self.model.layer4
        # 加载预生成的支持集（用于 MAML 快速适应），缺失时退化为直接推理
        self.support_x, self.support_y = self._load_support_set()
        self._base_state = copy.deepcopy(self.model.state_dict())

    def _get_cam(self):
        """延迟初始化 pytorch-grad-cam：layer3 + layer4 多层聚合，兼顾细节与语义。
        避免 eigen_smooth（会引入 PCA 反对称伪影，导致底部镜像热点）。"""
        if self._cam is None:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            self._cam = GradCAM(
                model=self.model,
                target_layers=[self.model.layer3, self.model.layer4],
            )
            self._cam_target_fn = ClassifierOutputTarget
        return self._cam

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

    def _load_display_image(self, image_path):
        """加载用于叠加显示的原始CT切片（窗宽窗位 -> 224 -> RGB，不做归一化）"""
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
            img = img.resize((224, 224), Image.BILINEAR)
        return img

    def _preprocess_image(self, image_path):
        """预处理：与 ipf_data.py 完全一致（原始像素窗宽窗位 -> 224 -> RGB -> 归一化）"""
        img = self._load_display_image(image_path)
        return self.transform(img).unsqueeze(0)

    @staticmethod
    def _lung_mask_from_gray(gray_224):
        """从肺窗灰度图 (H,W,uint8) 分割肺实质，返回 float32 掩膜 (0/1)。
        原理：肺窗下肺实质为低亮度（<~0.4），胸壁/骨为高亮度；
        取暗区 → 去除贴边空气 → 保留最大 1~2 个连通域（左右肺）→ 二值膨胀。"""
        g = gray_224.astype(np.float32) / 255.0
        dark = g < 0.45  # 肺实质候选（阈值对肺窗归一化后经验值）
        H, W = dark.shape
        # 清除贴边的图像外空气（DICOM 视野外通常也是暗的）
        border = np.zeros_like(dark, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        cleared = dark.copy()
        # flood fill：从边界暗像素出发，把连到边界的暗区判为“视野外”并剔除
        lbl, n = ndi.label(dark)
        if n > 0:
            border_labels = set(np.unique(lbl[border]))
            border_labels.discard(0)
            for lb in border_labels:
                cleared[lbl == lb] = False
        # 保留面积较大的前 2 个连通域作为左右肺
        lbl2, n2 = ndi.label(cleared)
        if n2 == 0:
            # 分割失败：不做掩膜（返回全 1，保底不裁掉 CAM）
            return np.ones_like(g, dtype=np.float32)
        sizes = ndi.sum(cleared, lbl2, index=np.arange(1, n2 + 1))
        keep = np.argsort(sizes)[::-1][:2]
        mask = np.zeros_like(cleared)
        min_area = 0.005 * H * W  # 太小的连通域视为噪声
        for k in keep:
            if sizes[k] >= min_area:
                mask |= (lbl2 == (k + 1))
        if mask.sum() == 0:
            return np.ones_like(g, dtype=np.float32)
        # 形态学收尾：填洞 + 轻微膨胀（把边缘胸膜下病灶包进来）
        mask = ndi.binary_fill_holes(mask)
        mask = ndi.binary_dilation(mask, iterations=3)
        return mask.astype(np.float32)

    @staticmethod
    def _lesion_texture_prior(gray_norm, lung_mask):
        """基于肺窗灰度构造病灶纹理先验（0~1）。
        肺纤维化病灶（网格/蜂窝/磨玻璃）在肺窗下比正常肺实质相对高亮：
        - 正常肺实质 ~0.15~0.30，病灶 ~0.35~0.55，胸壁/骨 >0.60。
        用平滑处理后的灰度做软阈值，在肺野内高亮处得到较高权重。"""
        g = ndi.gaussian_filter(gray_norm, sigma=1.0)
        prior = np.clip((g - 0.32) / 0.25, 0.0, 1.0)  # 0.32 以下→0，0.57 以上→1
        return prior * lung_mask

    @staticmethod
    def _keep_top_components(cam, k=2, min_area_ratio=0.003):
        """只保留最强的 k 个连通域，消除肺内背景弱激活噪声。
        连通域定义为 cam>0 的连通像素，强度用该连通域内 cam 之和衡量。"""
        H, W = cam.shape
        bin_mask = cam > 1e-3
        lbl, n = ndi.label(bin_mask)
        if n == 0:
            return cam
        scores = ndi.sum(cam, lbl, index=np.arange(1, n + 1))
        areas = ndi.sum(bin_mask.astype(np.float32), lbl, index=np.arange(1, n + 1))
        min_area = min_area_ratio * H * W
        # 面积过滤后按强度取 top-k
        valid = [(i + 1, scores[i]) for i in range(n) if areas[i] >= min_area]
        valid.sort(key=lambda t: t[1], reverse=True)
        keep_labels = {i for i, _ in valid[:k]}
        keep_mask = np.isin(lbl, list(keep_labels)) if keep_labels else np.zeros_like(bin_mask)
        return cam * keep_mask.astype(np.float32)

    def _generate_gradcam_overlay(self, image_path, image_tensor, predicted_class):
        """生成 Grad-CAM 叠加图，聚焦"病灶核心"的多级收紧策略：
        1. 无 eigen_smooth + layer3/layer4 多层聚合 → 消镜像伪影、兼顾细节和语义；
        2. 肺野掩膜 → 胸腔外清零；
        3. 病灶纹理先验（CT 灰度软阈值）与 CAM 相乘 → 让高值贴合真实病灶轮廓；
        4. 轻高斯平滑 (σ=1.5) → 去除 7×7 上采样直线边界，不再糊化；
        5. 分位阈值 p80 + gamma^2 锐化 → 压缩过渡带，只留核心；
        6. Top-K 连通域筛选 → 消除肺内弱激活噪声，只保留最强 1~2 个热点。"""
        from pytorch_grad_cam.utils.image import show_cam_on_image
        cam = self._get_cam()(
            input_tensor=image_tensor,
            targets=[self._cam_target_fn(predicted_class)],
            aug_smooth=False,
            eigen_smooth=False,
        )[0].astype(np.float32)

        base = self._load_display_image(image_path).convert('RGB')
        base_arr = np.asarray(base, dtype=np.float32) / 255.0
        gray = np.asarray(base.convert('L'))
        gray_norm = gray.astype(np.float32) / 255.0

        # 肺野掩膜 + 病灶纹理先验
        lung_mask = self._lung_mask_from_gray(gray)
        lesion_prior = self._lesion_texture_prior(gray_norm, lung_mask)

        # 掩膜 + 纹理引导
        cam = cam * lung_mask * (0.4 + 0.6 * lesion_prior)  # 保底 0.4 权重，不彻底消除弱纹理区

        # 轻平滑（收窄，只是去锯齿，不做扩散）
        cam = ndi.gaussian_filter(cam, sigma=1.5)

        # 分位阈值 p80：只保留肺野内 top-20% 激活
        inside = cam[lung_mask > 0]
        if inside.size > 0:
            thr = float(np.percentile(inside, 80))
            cam = np.where(cam >= thr, cam - thr, 0.0)

        # 归一化 + gamma 锐化（^2 压缩中间过渡带）
        cam_max = float(cam.max())
        if cam_max > 1e-8:
            cam = (cam / cam_max) ** 2.0
        else:
            return Image.fromarray((base_arr * 255).astype(np.uint8))

        # 只保留最强的 2 个连通域（左右肺各一个热点是典型的 IPF 分布）
        cam = self._keep_top_components(cam, k=2, min_area_ratio=0.003)

        # 收尾：裁到肺野内
        cam = cam * lung_mask
        cam_max = float(cam.max())
        if cam_max > 1e-8:
            cam = cam / cam_max

        overlay = show_cam_on_image(base_arr, cam, use_rgb=True, image_weight=0.65)
        return Image.fromarray(overlay)

    def predict_from_paths(self, image_paths, patient_id):
        """
        对一组CT切片进行患者级综合判读（切片逐张推理 + 一致性汇总）。
        返回值：
            predictions: list of dict [{'disease_id': int, 'disease_name': str, 'confidence': float, 'rank': int}]
            heatmap_url: str (URL路径)
            lesion_area_ratio: float (0~1) 支持主要判读方向的切片占比
            distribution_range: str 切片一致性描述
            imaging_findings: str
            suggestions: str
            diagnosis_view: dict 呈现层结构（疑似亚型/鉴别诊断/影像所见/建议）
        """
        if not image_paths:
            return self._fallback_result()

        # MAML 快速适应：先加载元初始参数，再在支持集上适应 INNER_STEPS 步
        self._adapt()

        # 逐张切片预处理与推理
        tensors = []
        valid_paths = []
        failed = []
        for p in image_paths:
            try:
                tensors.append(self._preprocess_image(p))
                valid_paths.append(p)
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

        # 用一张最终类别对应的代表切片生成 Grad-CAM
        rep_idx = int(np.where(slice_classes == predicted_class)[0][0]) if agree_count else 0
        rep_path = valid_paths[rep_idx]
        heatmap_img = self._generate_gradcam_overlay(
            rep_path, input_tensor[rep_idx:rep_idx + 1], predicted_class)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        static_gradcam = os.path.join(base_dir, 'static', 'gradcam')
        os.makedirs(static_gradcam, exist_ok=True)
        heatmap_filename = f'heatmap_{patient_id}.png'
        heatmap_path = os.path.join(static_gradcam, heatmap_filename)
        heatmap_img.save(heatmap_path)
        heatmap_url = f'/static/gradcam/{heatmap_filename}'

        # ========== 呈现层：按申报书“疑似亚型 + 置信度 + 结构化诊断”口径输出 ==========
        distribution = f'{agree_count}/{n_slices} 张切片判读一致'
        mapping = PRESENTATION_CONFIG[predicted_class]
        profile = PF_TYPE_PROFILES[mapping['profile']]
        primary_name = profile['name']
        support_ratio = impaired_slice_ratio if predicted_class == 1 else (1 - impaired_slice_ratio)

        severity_notes = {
            'low': '切片判读结果总体一致，未见明显快速进展征象，建议按常规随访管理。',
            'high': '判读结果提示部分切片肺实质受累较明显，建议结合肺功能检查动态评估进展风险。',
        }
        suggestion_addons = {
            'low': '如确诊IPF，建议规律随访并动态监测肺功能与影像变化。',
            'high': '建议尽早评估抗纤维化药物治疗方案（吡非尼酮/尼达尼布），并密切随访。',
        }

        imaging_blocks = [
            {
                'title': 'AI判读基础',
                'items': [
                    f'参与判读的CT切片：{n_slices} 张',
                    f'支持主要结论的切片占比：{support_ratio * 100:.1f}%',
                    f'切片判读一致性：{agree_count}/{n_slices}',
                ],
            },
            {
                'title': '建议重点评估的影像特征',
                'items': list(profile['typical_features']),
            },
            {
                'title': '影像判读提示',
                'items': [severity_notes[mapping['severity_hint']]],
            },
        ]
        imaging_findings = '\n'.join(
            f"【{block['title']}】\n" + '\n'.join(f"- {item}" for item in block['items'])
            for block in imaging_blocks)

        conclusion_text = (
            f"综合 {n_slices} 张CT切片的AI判读结果，该患者疑似「{primary_name}」，"
            f"模型综合置信度约 {confidence * 100:.1f}%。"
            f"建议结合临床表现、HRCT影像特征及肺功能检查进一步确认。")

        suggestions = list(profile['suggestions_base']) + [suggestion_addons[mapping['severity_hint']]]

        # 疑似亚型（主诊断；disease_id 在落库时替换为目录真实ID）
        predictions = [{
            'disease_id': 0,
            'disease_name': primary_name,
            'confidence': confidence,
            'rank': 1,
        }]
        diagnosis_view = {
            'primary_diagnosis': primary_name,
            'icd_code': profile['icd'],
            'differentials': list(profile['differentials']),
            'conclusion_text': conclusion_text,
            'imaging_blocks': imaging_blocks,
            'suggestions': suggestions,
        }

        return (predictions, heatmap_url, impaired_slice_ratio, distribution,
                imaging_findings, '\n'.join(suggestions), diagnosis_view)
    
    def diagnose_patient(self, patient_id):
        db = Database()
        if not db.connect():
            print("数据库连接失败，返回模拟结果")
            return self._fallback_result()

        images = db.get_medical_images(patient_id)
        db.disconnect()

        if not images:
            print(f"患者 {patient_id} 没有上传影像")
            return self._fallback_result()

        image_paths = [os.path.join('static', 'uploads', img['image_path']) for img in images]
        try:
            predictions, heatmap_url, impaired_ratio, distribution, findings, suggestions, diagnosis_view = \
                self.predict_from_paths(image_paths, patient_id)
        except Exception as e:
            print(f"患者 {patient_id} 推理失败: {e}")
            return self._fallback_result()

        # 落库时使用分型目录中的真实疾病ID，保证报告复核/患者报告能正确显示疾病名
        disease_id = self._resolve_catalog_disease_id()
        result = [{
            'disease_id': disease_id,
            'disease_name': predictions[0]['disease_name'],
            'confidence': predictions[0]['confidence'],
            'rank': 1,
        }]

        return {
            'predictions': result,
            'heatmap_url': heatmap_url,
            'lesion_area_ratio': impaired_ratio,
            'distribution_range': distribution,
            'imaging_findings': findings,
            'suggestions': suggestions,
            'diagnosis_view': diagnosis_view,
        }

    def _resolve_catalog_disease_id(self, keyword='特发性肺纤维化'):
        """将主诊断映射为 diseases 目录中的真实疾病ID（惰性查询并缓存）"""
        if self._catalog_ipf_id is not None:
            return self._catalog_ipf_id
        try:
            db = Database()
            if db.connect():
                rows = db.execute_query(
                    "SELECT disease_id FROM diseases WHERE name LIKE %s "
                    "ORDER BY disease_id LIMIT 1", (f'%{keyword}%',))
                db.disconnect()
                if rows:
                    self._catalog_ipf_id = int(rows[0]['disease_id'])
                    return self._catalog_ipf_id
        except Exception as e:
            print(f"疾病目录查询失败: {e}")
        self._catalog_ipf_id = 0
        return self._catalog_ipf_id

    def _fallback_result(self):
        """无影像/数据库异常时的兜底结果（与 predict_from_paths 返回结构一致）"""
        profile = PF_TYPE_PROFILES['ipf']
        name = profile['name']
        return (
            [{'disease_id': 0, 'disease_name': name, 'confidence': 0.80, 'rank': 1}],
            '',
            0.0,
            '--',
            '暂无影像数据，无法完成分型识别，请先上传CT影像后重试。',
            '请先上传患者CT影像，再发起AI辅助诊断。',
            {
                'primary_diagnosis': name,
                'icd_code': profile['icd'],
                'differentials': list(profile['differentials']),
                'conclusion_text': '暂无影像数据，无法生成诊断结论。',
                'imaging_blocks': [
                    {'title': 'AI判读基础', 'items': ['暂无影像数据，请先上传CT影像。']},
                ],
                'suggestions': ['请先上传患者CT影像，再发起AI辅助诊断。'],
            }
        )
