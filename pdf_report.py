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


# ---------- 字体注册（Windows / Linux 中文字体自动识别） ----------
def _register_fonts():
    # Windows 常用中文字体
    for name, path in [
        ('SimHei', r'C:\Windows\Fonts\simhei.ttf'),
        ('Deng', r'C:\Windows\Fonts\Deng.ttf'),
        ('DengB', r'C:\Windows\Fonts\Dengb.ttf'),
    ]:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception:
                pass
    # Linux 常用中文字体：.ttc 逐个 subfont 尝试，.ttf 直接注册
    for name, path in [
        ('NotoSansCJK', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        ('NotoSansCJKB', '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'),
        ('DroidSansFallback', '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
        ('WenQuanYi', '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'),
        ('WenQuanYiB', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'),
    ]:
        if not os.path.exists(path):
            continue
        if path.endswith('.ttf'):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                continue
            except Exception:
                pass
        for idx in range(8):
            try:
                pdfmetrics.registerFont(TTFont(name, path, subfontIndex=idx))
                break
            except Exception:
                continue
    # 字体映射（缺字回退）
    pdfmetrics.registerFontFamily(
        'Deng', normal='Deng', bold='DengB', italic='Deng', boldItalic='DengB')
    pdfmetrics.registerFontFamily(
        'NotoSansCJK', normal='NotoSansCJK', bold='NotoSansCJKB',
        italic='NotoSansCJK', boldItalic='NotoSansCJKB')


_register_fonts()

_REG = set(pdfmetrics.getRegisteredFontNames())


def _pick_font(prefs):
    """从候选字体中选第一个已注册的；全部缺失时退回 Helvetica（避免崩溃）"""
    for n in prefs:
        if n in _REG:
            return n
    return 'Helvetica'


HEAD_FONT = _pick_font(['SimHei', 'NotoSansCJK', 'Deng', 'WenQuanYi', 'DroidSansFallback'])
BODY_FONT = _pick_font(['Deng', 'NotoSansCJK', 'SimHei', 'WenQuanYi', 'DroidSansFallback'])
BOLD_FONT = _pick_font(['DengB', 'NotoSansCJKB', 'NotoSansCJK', 'SimHei',
                        'WenQuanYiB', 'WenQuanYi', 'DroidSansFallback'])

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


def _differential_table(st, differentials):
    """鉴别诊断参考表：候选亚型 + 鉴别要点"""
    header = [Paragraph('候选亚型', st['cell_b']),
              Paragraph('鉴别要点', st['cell_b'])]
    rows = [header]
    for d in differentials:
        name = d.get('name', '') if isinstance(d, dict) else str(d)
        note = d.get('note', '') if isinstance(d, dict) else ''
        rows.append([Paragraph(name, st['cell']), Paragraph(note or '需结合临床资料进一步鉴别', st['cell'])])
    t = Table(rows, colWidths=[58 * mm, 119 * mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, ROW_ALT]),
    ]))
    return t


def _slice_stat_table(st, n_slices, support_ratio, distribution):
    rows = [
        [Paragraph('参与判读的切片数', st['cell']), Paragraph(str(n_slices), st['cell_b'])],
        [Paragraph('支持主要结论的切片占比', st['cell']),
         Paragraph(f'{support_ratio * 100:.1f}%', st['cell_b'])],
        [Paragraph('切片判读一致性', st['cell']), Paragraph(distribution or '--', st['cell'])],
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


def _bullet_flowables(st, items):
    """有序列表 Flowable 列表"""
    flow = []
    for i, item in enumerate(items, 1):
        flow.append(Paragraph(f'{i}. {item}', ParagraphStyle(
            'bullet', fontName=BODY_FONT, fontSize=10, leading=16,
            textColor=colors.HexColor('#374151'), spaceAfter=2)))
    return flow


def _imaging_blocks_flowables(st, blocks):
    """影像所见结构化块（标题 + 列表）"""
    flow = []
    for block in blocks:
        title = block.get('title', '')
        items = block.get('items', [])
        if title:
            flow.append(Paragraph(title, ParagraphStyle(
                'ib', fontName=BOLD_FONT, fontSize=10.5, leading=15,
                textColor=PRIMARY_DARK, spaceBefore=4, spaceAfter=2)))
        for item in items:
            flow.append(Paragraph(f'· {item}', ParagraphStyle(
                'ib_item', fontName=BODY_FONT, fontSize=10, leading=15,
                textColor=colors.HexColor('#374151'), leftIndent=4 * mm, spaceAfter=1)))
    return flow


def build_diagnosis_report_pdf(data, doctor_name='', patient_name='') -> bytes:
    """根据诊断结果数据生成 A4 报告 PDF，返回字节流。
    data 字段: patient_id, predictions, primary_diagnosis, icd_code, differentials,
              conclusion_text, imaging_blocks, suggestions_list, imaging_findings,
              suggestions, lesion_area_ratio, distribution_range, heatmap_url,
              time_cost, model_version, n_slices
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
        canvas.drawString(18 * mm, A4[1] - 5.6 * mm, '肺影智诊 · 肺纤维化临床决策支持系统')
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
    story.append(Paragraph('肺纤维化分型辅助识别 · 基于 MAML 元迁移学习的 CT 影像小样本分类', st['subtitle']))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(f'报告编号：{report_no}　|　生成时间：{now.strftime("%Y-%m-%d %H:%M")}', st['subtitle']))
    story.append(Spacer(1, 5 * mm))

    # 患者与检查信息
    story.append(Paragraph('一、患者与检查信息', st['h2']))
    story.append(_meta_table(st, [
        ('患者ID', patient_id or '--'),
        ('患者姓名', patient_name or '--'),
        ('检查类型', '胸部 CT 影像（DICOM 序列）'),
        ('诊断模型', 'MAML 元迁移学习（ResNet-18 骨干，肺纤维化分型辅助识别）'),
        ('模型版本', model_version),
        ('诊断耗时', f"{data.get('time_cost', '--')} 秒"),
    ]))

    # AI 诊断结论（疑似亚型 + 综合置信度）
    story.append(Paragraph('二、AI 诊断结论', st['h2']))
    preds = data.get('predictions') or []
    if preds:
        winner = max(preds, key=lambda p: p.get('confidence', 0))
        conclusion = data.get('conclusion_text') or (
            f'综合 {data.get("n_slices", "--")} 张CT切片的AI判读结果，'
            f'该患者疑似「{winner.get("disease_name", "")}」，'
            f'模型综合置信度约 {float(winner.get("confidence", 0)) * 100:.1f}%。'
            f'建议结合临床表现、HRCT影像特征及肺功能检查进一步确认。')
        story.append(_conclusion_box(st, conclusion))
        conf = float(winner.get('confidence', 0))
        story.append(Spacer(1, 2 * mm))
        cb = ConfidenceBar(conf, width=120 * mm, height=6 * mm)
        cb.hAlign = 'LEFT'
        story.append(cb)
        story.append(Spacer(1, 1 * mm))
        story.append(Paragraph(f'综合置信度：{conf * 100:.1f}%', st['meta']))

    # 鉴别诊断参考
    diffs = data.get('differentials') or []
    if diffs:
        story.append(Paragraph('三、鉴别诊断参考', st['h2']))
        story.append(_differential_table(st, diffs))

    # 影像所见（AI判读）
    story.append(Paragraph('四、影像所见（AI判读）', st['h2']))
    blocks = data.get('imaging_blocks') or []
    if blocks:
        story.extend(_imaging_blocks_flowables(st, blocks))
    else:
        story.append(Paragraph(data.get('imaging_findings') or '--', st['body']))

    # 切片级统计
    story.append(Paragraph('五、切片级统计', st['h2']))
    story.append(_slice_stat_table(
        st,
        data.get('n_slices', '--'),
        float(data.get('lesion_area_ratio', 0) or 0),
        data.get('distribution_range') or '--'
    ))

    # 检查建议
    story.append(Paragraph('六、检查建议', st['h2']))
    sugg_list = data.get('suggestions_list') or []
    if sugg_list:
        story.extend(_bullet_flowables(st, sugg_list))
    else:
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
                '图：Grad-CAM 热力图半透明叠加于原始 CT 切片之上，'
                '高亮区域表示对AI判读结果贡献较大的肺实质区域。',
                ParagraphStyle('cap', fontName=BODY_FONT, fontSize=8.5, leading=12,
                               textColor=GRAY, alignment=TA_CENTER)))

    # 附注
    story.append(Paragraph('八、附注', st['h2']))
    notes = [
        '1. 本报告由 AI 模型基于肺纤维化 CT 影像自动生成，诊断结论经切片级判读与一致性汇总得到，'
        '仅供临床参考。',
        '2. 模型方法：MAML 元迁移学习（ResNet-18 骨干网络），面向小样本CT影像场景的'
        '肺纤维化分型辅助识别。',
        '3. 鉴别诊断类型为临床复核提示，不构成概率排序；最终诊断需由执业医师结合'
        '临床表现、肺功能检查及其他影像资料综合确认。',
        '4. 本报告不构成最终诊断意见，'
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
            {'disease_name': '特发性肺纤维化（IPF）', 'confidence': 0.9424},
        ],
        'primary_diagnosis': '特发性肺纤维化（IPF）',
        'icd_code': 'J84.1',
        'differentials': [
            {'name': '非特异性间质性肺炎（NSIP）', 'note': '常伴较广泛的磨玻璃影，需结合HRCT分布鉴别'},
            {'name': '慢性过敏性肺炎（HP）', 'note': '与吸入性抗原暴露相关，需追问暴露史'},
            {'name': '结缔组织病相关ILD（CTD-ILD）', 'note': '需结合自身抗体及系统表现筛查'},
        ],
        'conclusion_text': '综合 28 张CT切片的AI判读结果，该患者疑似「特发性肺纤维化（IPF）」，模型综合置信度约 94.2%。建议结合临床表现、HRCT影像特征及肺功能检查进一步确认。',
        'imaging_blocks': [
            {'title': 'AI判读基础', 'items': ['参与判读的CT切片：28 张', '支持主要结论的切片占比：100.0%', '切片判读一致性：28/28']},
            {'title': '建议重点评估的影像特征', 'items': ['双肺底及胸膜下分布为主的网格影', '牵拉性支气管扩张', '蜂窝影（中晚期常见）', '磨玻璃影通常少见且范围局限']},
            {'title': '影像判读提示', 'items': ['切片判读结果总体一致，未见明显快速进展征象，建议按常规随访管理。']},
        ],
        'suggestions_list': [
            '建议完善肺功能检查（FVC%、DLCO）评估功能受损程度',
            '建议高分辨率CT（HRCT）复查，进一步明确影像分型',
            '结合临床与影像资料，必要时行多学科（呼吸/影像/病理）讨论',
            '如确诊IPF，建议规律随访并动态监测肺功能与影像变化。',
        ],
        'n_slices': 28,
        'lesion_area_ratio': 0.0,
        'distribution_range': '28/28 张切片判读一致',
        'imaging_findings': '【AI判读基础】\n- 参与判读的CT切片：28 张\n- 支持主要结论的切片占比：100.0%',
        'suggestions': '建议完善肺功能检查（FVC%、DLCO）评估功能受损程度。',
        'heatmap_url': '/static/gradcam/heatmap_6.png',
        'time_cost': 38.5,
        'model_version': 'best_maml_fold1.pth',
    }
    out = 'tmp_sample_report.pdf'
    os.makedirs('tmp', exist_ok=True)
    with open(out, 'wb') as f:
        f.write(build_diagnosis_report_pdf(sample, doctor_name='王医生', patient_name='小王'))
    print('written', out)
