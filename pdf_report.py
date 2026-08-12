"""AI 辅助诊断报告 PDF 生成（reportlab，A4 中文规范排版）"""
import os
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, Image, PageTemplate,
    Paragraph, Spacer, Table, TableStyle
)


# ---------- 字体注册（Windows 中文字体） ----------
def _register_fonts():
    font_dir = r'C:\Windows\Fonts'
    candidates = {
        'SimHei': ('simhei.ttf', None),
        'Deng': ('Deng.ttf', None),
        'DengB': ('Dengb.ttf', None),
    }
    for name, (file, _) in candidates.items():
        path = os.path.join(font_dir, file)
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(name, path))
    # 字体映射（缺字回退）
    pdfmetrics.registerFontFamily(
        'Deng', normal='Deng', bold='DengB', italic='Deng', boldItalic='DengB')


_register_fonts()

HEAD_FONT = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Deng'
BODY_FONT = 'Deng' if 'Deng' in pdfmetrics.getRegisteredFontNames() else 'SimHei'
BOLD_FONT = 'DengB' if 'DengB' in pdfmetrics.getRegisteredFontNames() else HEAD_FONT

PRIMARY = colors.HexColor('#165DFF')
PRIMARY_DARK = colors.HexColor('#0E42D2')
HEADER_BG = colors.HexColor('#EAF1FF')
ROW_ALT = colors.HexColor('#F7FAFF')
BORDER = colors.HexColor('#D9E1EC')
GRAY = colors.HexColor('#6B7280')


# ---------- 样式 ----------
def _styles():
    return {
        'title': ParagraphStyle('title', fontName=HEAD_FONT, fontSize=22, leading=30,
                                alignment=TA_CENTER, textColor=PRIMARY_DARK),
        'subtitle': ParagraphStyle('subtitle', fontName=BODY_FONT, fontSize=10.5, leading=15,
                                   alignment=TA_CENTER, textColor=GRAY),
        'h2': ParagraphStyle('h2', fontName=HEAD_FONT, fontSize=12.5, leading=17,
                             textColor=PRIMARY_DARK, spaceBefore=12, spaceAfter=5,
                             keepWithNext=1),
        'body': ParagraphStyle('body', fontName=BODY_FONT, fontSize=10.5, leading=17,
                               alignment=TA_LEFT),
        'meta': ParagraphStyle('meta', fontName=BODY_FONT, fontSize=9.5, leading=15,
                               textColor=colors.HexColor('#374151')),
        'cell': ParagraphStyle('cell', fontName=BODY_FONT, fontSize=10.5, leading=15),
        'cell_b': ParagraphStyle('cell_b', fontName=BOLD_FONT, fontSize=10.5, leading=15),
        'footer': ParagraphStyle('footer', fontName=BODY_FONT, fontSize=8, leading=11,
                                 textColor=GRAY),
    }


class ConfidenceBar(Flowable):
    """置信度条：背景 + 填充 + 百分比文本"""
    def __init__(self, ratio, width=70 * mm, height=7 * mm, color=PRIMARY):
        super().__init__()
        self.ratio = max(0.0, min(1.0, float(ratio)))
        self.width = width
        self.height = height
        self.color = color

    def wrap(self, availWidth, availHeight):
        return self.width, self.height

    def draw(self):
        c = self.canv
        r = self.height / 2
        # 背景
        c.setFillColor(colors.HexColor('#E9EEF5'))
        c.roundRect(0, 0, self.width, self.height, r, stroke=0, fill=1)
        # 填充
        fill_w = self.width * self.ratio
        if fill_w > r * 2:
            c.setFillColor(self.color)
            c.roundRect(0, 0, fill_w, self.height, r, stroke=0, fill=1)
        # 文本
        c.setFillColor(colors.white if self.ratio > 0.5 else colors.HexColor('#374151'))
        c.setFont(BODY_FONT, 8)
        c.drawCentredString(self.width / 2, (self.height - 9) / 2,
                            f'{self.ratio * 100:.1f}%')


def _meta_table(st, meta_rows):
    data = [[Paragraph(k, st['meta']), Paragraph(v, st['meta'])] for k, v in meta_rows]
    t = Table(data, colWidths=[32 * mm, 145 * mm])
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, ROW_ALT]),
        ('LINEBELOW', (0, 0), (-1, -2), 0.4, BORDER),
    ]))
    return t


def _conclusion_box(st, conclusion_text):
    data = [[Paragraph(conclusion_text, ParagraphStyle(
        'concl', fontName=BOLD_FONT, fontSize=11.5, leading=18,
        textColor=colors.HexColor('#0B3D91')))]]
    t = Table(data, colWidths=[177 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HEADER_BG),
        ('BOX', (0, 0), (-1, -1), 0.8, PRIMARY),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
    ]))
    return t


def _prediction_table(st, predictions, n_slices):
    header = [Paragraph('类别', st['cell_b']),
              Paragraph('切片平均置信度', st['cell_b']),
              Paragraph('置信度分布', st['cell_b'])]
    rows = [header]
    for p in predictions:
        name = p.get('disease_name', '')
        conf = float(p.get('confidence', 0))
        rows.append([Paragraph(name, st['cell']),
                     Paragraph(f'{conf * 100:.2f}%', st['cell']),
                     ConfidenceBar(conf)])
    t = Table(rows, colWidths=[62 * mm, 34 * mm, 81 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 1), (1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT]),
    ]))
    return t


def _slice_stat_table(st, n_slices, impaired_ratio, distribution):
    rows = [
        [Paragraph('参与诊断的切片数', st['cell']), Paragraph(str(n_slices), st['cell_b'])],
        [Paragraph('严重受损类切片占比', st['cell']),
         Paragraph(f'{impaired_ratio * 100:.1f}%', st['cell_b'])],
        [Paragraph('切片投票一致性', st['cell']), Paragraph(distribution or '--', st['cell'])],
    ]
    t = Table(rows, colWidths=[62 * mm, 115 * mm])
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('BACKGROUND', (0, 0), (0, -1), ROW_ALT),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def build_diagnosis_report_pdf(data, doctor_name='', patient_name='') -> bytes:
    """根据诊断结果数据生成 A4 报告 PDF，返回字节流。
    data 字段: patient_id, predictions, lesion_area_ratio, distribution_range,
              imaging_findings, suggestions, heatmap_url, time_cost, model_version
    """
    st = _styles()
    buf = BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=20 * mm,
        title='肺影智诊 AI辅助诊断报告', author='肺影智诊'
    )

    patient_id = str(data.get('patient_id', ''))
    now = datetime.now()
    report_no = f'PF-{patient_id or "X"}-{now.strftime("%Y%m%d%H%M")}'
    model_version = data.get('model_version') or 'best_maml_fold1.pth'

    def on_page(canvas, doc_):
        canvas.saveState()
        # 顶部品牌条
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont(HEAD_FONT, 11)
        canvas.drawString(18 * mm, A4[1] - 5.6 * mm, '肺影智诊 · 罕见病临床决策支持系统')
        # 底部页脚
        canvas.setFillColor(GRAY)
        canvas.setFont(BODY_FONT, 8)
        canvas.drawString(18 * mm, 11 * mm,
                          '本报告由 AI 模型自动生成，仅供临床参考，最终诊断请由执业医师结合临床资料复核确认。')
        canvas.drawRightString(A4[0] - 18 * mm, 11 * mm, f'第 {doc_.page} 页')
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
        canvas.restoreState()

    frame = Frame(18 * mm, 20 * mm, A4[0] - 36 * mm, A4[1] - 36 * mm, id='main')
    doc.addPageTemplates([PageTemplate(id='page', frames=[frame], onPage=on_page)])

    story = []
    # 标题区
    story.append(Paragraph('AI 辅助诊断报告', st['title']))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph('IPF 预后分型 · 基于 MAML 元迁移学习的 CT 影像小样本分类', st['subtitle']))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(f'报告编号：{report_no}　|　生成时间：{now.strftime("%Y-%m-%d %H:%M")}', st['subtitle']))
    story.append(Spacer(1, 5 * mm))

    # 患者与检查信息
    story.append(Paragraph('一、患者与检查信息', st['h2']))
    story.append(_meta_table(st, [
        ('患者ID', patient_id or '--'),
        ('患者姓名', patient_name or '--'),
        ('检查类型', '胸部 CT 影像（DICOM 序列）'),
        ('诊断模型', f'MAML 元迁移学习（ResNet-18 骨干，2-way 2-shot）'),
        ('模型版本', model_version),
        ('诊断耗时', f"{data.get('time_cost', '--')} 秒"),
    ]))

    # 诊断结论
    story.append(Paragraph('二、诊断结论（患者级多数投票）', st['h2']))
    preds = data.get('predictions') or []
    if preds:
        winner = max(preds, key=lambda p: p.get('confidence', 0))
        conclusion = (f'综合 {data.get("n_slices", "--")} 张 CT 切片的患者级多数投票，'
                      f'该患者被判定为「{winner.get("disease_name", "")}」，'
                      f'模型置信度约 {float(winner.get("confidence", 0)) * 100:.1f}%。')
        story.append(_conclusion_box(st, conclusion))

    # 分类置信度
    story.append(Paragraph('三、分类置信度', st['h2']))
    story.append(_prediction_table(st, preds, data.get('n_slices')))

    # 切片级统计
    story.append(Paragraph('四、切片级统计', st['h2']))
    story.append(_slice_stat_table(
        st,
        data.get('n_slices', '--'),
        float(data.get('lesion_area_ratio', 0) or 0),
        data.get('distribution_range') or '--'
    ))

    # 影像所见
    story.append(Paragraph('五、影像所见', st['h2']))
    story.append(Paragraph(data.get('imaging_findings') or '--', st['body']))

    # 检查建议
    story.append(Paragraph('六、检查建议', st['h2']))
    story.append(Paragraph(data.get('suggestions') or '--', st['body']))

    # Grad-CAM 热力图
    heatmap_url = data.get('heatmap_url') or ''
    if heatmap_url:
        base = os.path.dirname(os.path.abspath(__file__))
        heat_path = os.path.normpath(os.path.join(base, heatmap_url.lstrip('/')))
        if os.path.exists(heat_path):
            story.append(Paragraph('七、Grad-CAM 注意力热力图', st['h2']))
            img = Image(heat_path, width=82 * mm, height=82 * mm)
            img.hAlign = 'CENTER'
            story.append(img)
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                '图：模型重点关注区域的 Grad-CAM 可视化（红色区域表示对分类决策贡献较大的像素）。',
                ParagraphStyle('cap', fontName=BODY_FONT, fontSize=8.5, leading=12,
                               textColor=GRAY, alignment=TA_CENTER)))

    # 附注
    story.append(Paragraph('八、附注', st['h2']))
    notes = [
        '1. 分类标签定义：类0 为相对稳定组（Percent≥90），类1 为严重受损组（Percent≤65），'
        '依据公开 OSIC 肺纤维化数据集的 FVC 百分比指标划分。',
        '2. 模型方法：MAML 元迁移学习，ResNet-18 骨干网络（ImageNet 预训练），'
        '2-way 2-shot 患者级任务构造，5 折患者级分层交叉验证；诊断结论由切片级预测经患者级多数投票得到。',
        '3. 本报告由 AI 模型自动生成，仅供临床参考，不构成最终诊断意见；'
        '请执业医师结合患者临床表现、肺功能检查及其他影像资料综合判读。',
    ]
    for n in notes:
        story.append(Paragraph(n, ParagraphStyle(
            'note', fontName=BODY_FONT, fontSize=9.5, leading=15,
            textColor=colors.HexColor('#374151'), spaceAfter=3)))

    # 签名区
    story.append(Spacer(1, 6 * mm))
    sig = Table(
        [[Paragraph(f'诊断医师：{doctor_name or "__________"}', st['meta']),
          Paragraph(f'复核日期：____________', st['meta'])]],
        colWidths=[88 * mm, 89 * mm]
    )
    sig.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sig)

    doc.build(story)
    return buf.getvalue()


if __name__ == '__main__':
    # 冒烟测试：生成示例报告
    sample = {
        'patient_id': '6',
        'predictions': [
            {'disease_name': '相对稳定组 (Percent≥90)', 'confidence': 0.9424},
            {'disease_name': '严重受损组 (Percent≤65)', 'confidence': 0.0576},
        ],
        'n_slices': 28,
        'lesion_area_ratio': 0.0,
        'distribution_range': '28/28 张切片支持该结论',
        'imaging_findings': '基于 28 张CT切片的患者级多数投票，模型判定为「相对稳定组 (Percent≥90)」，提示肺功能相对稳定表型可能性较高。',
        'suggestions': '建议维持常规随访，定期复查肺功能与CT影像，监测FVC变化趋势。',
        'heatmap_url': '/static/gradcam/heatmap_6.png',
        'time_cost': 32,
    }
    out = 'tmp_sample_report.pdf'
    os.makedirs('tmp', exist_ok=True)
    with open(out, 'wb') as f:
        f.write(build_diagnosis_report_pdf(sample, doctor_name='王医生', patient_name='小王'))
    print('written', out)
