from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, flash, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import uuid
import io
import re
import json
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from database import Database
from pf_diagnosis_service import PFDianosisService
import datetime
import time
from functools import wraps
import pydicom
from PIL import Image
import numpy as np
from decimal import Decimal


app = Flask(__name__)
app.config.from_object(Config)
app.config['SECRET_KEY'] = '4a55ca79078c8fcd2435348e794f8d86d9fd64c6feef9c752ec2a6dc0c69c22c'

# 初始化Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


def admin_required(f):
    """管理员权限装饰器：仅 users.role = 'admin' 可访问"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if current_user.user_type != 'admin':
            flash('无权访问此页面', 'danger')
            return redirect(url_for('doctor_dashboard'))
        return f(*args, **kwargs)
    return wrapper

# 用户类
class User(UserMixin):
    def __init__(self, user_id, username, user_type, full_name):
        self.id = user_id
        self.username = username
        self.user_type = user_type
        self.full_name = full_name

# 用户加载器
@login_manager.user_loader
def load_user(user_id):
    db = Database()
    if not db.connect():
        return None
    
    user_data = db.execute_query("SELECT * FROM users WHERE user_id = %s", (user_id,))
    db.disconnect()
    
    if user_data:
        user = user_data[0]
        return User(user['user_id'], user['username'], user['role'], user['full_name'])
    return None

# 初始化肺纤维化诊断服务（全局单例）
import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'best_maml_fold1.pth')
pf_service = PFDianosisService(model_path=MODEL_PATH)

# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        db = Database()
        if not db.connect():
            flash('数据库连接失败', 'danger')
            return render_template('login.html')

        user_data = db.execute_query("SELECT * FROM users WHERE username = %s", (username,))
        if not user_data:
            db.disconnect()
            flash('用户名不存在', 'danger')
            return render_template('login.html')

        user = user_data[0]
        if not check_password_hash(user['password_hash'], password):
            db.disconnect()
            flash('密码错误', 'danger')
            return render_template('login.html')

        # 登录成功，创建 session
        login_user(User(user['user_id'], user['username'], user['role'], user['full_name']), remember=remember)

        # 更新最后登录时间（如果表中没有该列，请先添加，或注释下行）
        db.execute_insert("UPDATE users SET last_login = NOW() WHERE user_id = %s", (user['user_id'],))
        db.disconnect()

        # 跳转：管理员 / 医生 / 患者 分流
        if user['role'] == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif user['role'] == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))

    return render_template('login.html')

# 注册路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        user_type = request.form.get('user_type')
        
        if not username or not password or not email or not full_name:
            flash('请填写所有必填字段', 'danger')
            return render_template('register.html')
        
        if not user_type or user_type not in ['doctor', 'patient']:
            flash('请选择有效的身份', 'danger')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('密码不匹配', 'danger')
            return render_template('register.html')
        
        db = Database()
        if not db.connect():
            flash('数据库连接失败', 'danger')
            return render_template('register.html')
        
        existing_user = db.execute_query("SELECT * FROM users WHERE username = %s", (username,))
        if existing_user:
            flash('用户名已存在', 'danger')
            db.disconnect()
            return render_template('register.html')
        
        existing_email = db.execute_query("SELECT * FROM users WHERE email = %s", (email,))
        if existing_email:
            flash('邮箱已存在', 'danger')
            db.disconnect()
            return render_template('register.html')
        
        hashed_password = generate_password_hash(password)
        user_id = db.execute_insert(
            "INSERT INTO users (username, password_hash, email, full_name, role) VALUES (%s, %s, %s, %s, %s)",
            (username, hashed_password, email, full_name, user_type)
        )
        
        if user_id is None:
            db.disconnect()
            flash('注册失败，请重试', 'danger')
            return render_template('register.html')
        
        if user_type == 'patient':
            patient_name = request.form.get('patient_name') or full_name
            patient_gender = request.form.get('patient_gender')
            patient_age = request.form.get('patient_age')
            patient_insert_result = db.execute_insert(
                "INSERT INTO patients (name, age, gender, user_id) VALUES (%s, %s, %s, %s)",
                (patient_name, patient_age, patient_gender, user_id)
            )
            if patient_insert_result is None:
                # 如果患者信息插入失败，可以选择删除用户或只是警告
                flash('注册成功，但患者信息保存失败，请联系管理员', 'warning')
        
        db.disconnect()
        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# 我的报告（患者）
@app.route('/my_reports')
@login_required
def my_reports():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('my_reports.html', reports=[],
                               stats={'total': 0, 'pending': 0, 'completed': 0, 'reviewed': 0})
    
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到关联的患者信息', 'danger')
        return render_template('my_reports.html', reports=[],
                               stats={'total': 0, 'pending': 0, 'completed': 0, 'reviewed': 0})
    
    patient_id = patient_data[0]['patient_id']
    reports = db.get_diagnosis_reports(patient_id) or []
    for r in reports:
        r['predictions'] = db.get_disease_predictions(r['report_id']) or []
        for f in ('created_at', 'reviewed_at'):
            if hasattr(r.get(f), 'strftime'):
                r[f] = r[f].strftime('%Y-%m-%d %H:%M')
        if isinstance(r.get('lesion_area_ratio'), Decimal):
            r['lesion_area_ratio'] = float(r['lesion_area_ratio'])
    stats = {
        'total': len(reports),
        'pending': sum(1 for r in reports if r.get('status') == 'pending'),
        'completed': sum(1 for r in reports if r.get('status') == 'completed'),
        'reviewed': sum(1 for r in reports if r.get('status') == 'reviewed'),
    }
    db.disconnect()
    return render_template('my_reports.html', reports=reports, stats=stats)


def _parse_findings_blocks(text):
    """把影像所见文本解析为结构化块（【标题】 + 条目）"""
    blocks = []
    cur = None
    for line in (text or '').splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('【') and '】' in line:
            cur = {'title': line.strip('【】'), 'lines': []}
            blocks.append(cur)
        elif cur is not None:
            cur['lines'].append(line.lstrip('- '))
    return blocks


# 患者报告详情（规范报告）
@app.route('/patient/report/<int:report_id>')
@login_required
def patient_report_detail(report_id):
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return redirect(url_for('my_reports'))
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id=%s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到患者信息', 'danger')
        return redirect(url_for('my_reports'))
    patient = patient_data[0]
    report_rows = db.get_report_by_id(report_id) or []
    if not report_rows or report_rows[0]['patient_id'] != patient['patient_id']:
        db.disconnect()
        flash('报告不存在或无权查看', 'danger')
        return redirect(url_for('my_reports'))
    report = report_rows[0]
    predictions = db.get_disease_predictions(report_id) or []
    images = db.get_medical_images(patient['patient_id']) or []
    db.disconnect()

    for f in ('created_at', 'reviewed_at'):
        if hasattr(report.get(f), 'strftime'):
            report[f] = report[f].strftime('%Y-%m-%d %H:%M')
    if isinstance(report.get('lesion_area_ratio'), Decimal):
        report['lesion_area_ratio'] = float(report['lesion_area_ratio'])
    for p in predictions:
        p['confidence'] = float(p.get('confidence') or 0)
    for img in images:
        if hasattr(img.get('upload_date'), 'strftime'):
            img['upload_date'] = img['upload_date'].strftime('%Y-%m-%d')

    findings_blocks = _parse_findings_blocks(report.get('findings'))
    suggestions = [s for s in (report.get('suggestions') or '').splitlines() if s.strip()]
    differentials = []
    try:
        diffs = json.loads(report.get('differentials') or '[]')
        if isinstance(diffs, list):
            differentials = diffs
    except Exception:
        differentials = []

    heatmap_url = None
    hp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'static', 'gradcam', f'heatmap_{patient["patient_id"]}.png')
    if os.path.exists(hp):
        heatmap_url = f'/static/gradcam/heatmap_{patient["patient_id"]}.png'

    return render_template('patient_report_detail.html', report=report, predictions=predictions,
                           images=images, findings_blocks=findings_blocks, suggestions=suggestions,
                           differentials=differentials, heatmap_url=heatmap_url, patient=patient)

# 退出登录
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('您已成功退出登录', 'success')
    return redirect(url_for('login'))

# 医生仪表板
@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    model_trained = os.path.exists(MODEL_PATH)
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('doctor_dashboard.html', patients=[], model_trained=model_trained)
    
    patients = db.get_patients() or []
    
    # 统计数据
    total_patients = len(patients)
    today = datetime.date.today().strftime('%Y-%m-%d')
    today_appointments = db.execute_query(
        "SELECT COUNT(*) as count FROM appointments WHERE appointment_date = %s AND status != 'cancelled'",
        (today,)
    )
    today_appointments = today_appointments[0]['count'] if today_appointments else 0
    
    pending_reports = db.execute_query(
        "SELECT COUNT(*) as count FROM diagnosis_reports WHERE status = 'pending'"
    )
    pending_reports = pending_reports[0]['count'] if pending_reports else 0
    
    unread_chat = db.get_total_unread_chat_count(current_user.id)
    unread_notifications = db.get_unread_notification_count(current_user.id)
    
    # 患者年龄/性别分布（用于图表）
    age_groups = {'0-18': 0, '19-40': 0, '41-60': 0, '60+': 0}
    gender_counts = {'male': 0, 'female': 0, 'other': 0}
    for p in patients:
        gender = p.get('gender') or 'other'
        if gender in gender_counts:
            gender_counts[gender] += 1
        else:
            gender_counts['other'] += 1
        age = p.get('age')
        if age is not None:
            try:
                age = int(age)
            except (ValueError, TypeError):
                age = None
        if age is None:
            age_groups['60+'] += 1
        elif age <= 18:
            age_groups['0-18'] += 1
        elif age <= 40:
            age_groups['19-40'] += 1
        elif age <= 60:
            age_groups['41-60'] += 1
        else:
            age_groups['60+'] += 1
    
    db.disconnect()
    return render_template('doctor_dashboard.html',
                         patients=patients,
                         model_trained=model_trained,
                         total_patients=total_patients,
                         today_appointments=today_appointments,
                         pending_reports=pending_reports,
                         unread_chat=unread_chat,
                         unread_notifications=unread_notifications,
                         age_groups=age_groups,
                         gender_counts=gender_counts,
                         today=today)

# 患者仪表板
@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_dashboard.html', reports=[], patient=None)
    
    patient_data = db.execute_query(
        "SELECT * FROM patients WHERE user_id = %s AND is_deleted = 0", (current_user.id,)
    )
    if not patient_data:
        db.disconnect()
        flash('未找到关联的患者信息', 'danger')
        return render_template('patient_dashboard.html', reports=[], patient=None)
    
    patient_id = patient_data[0]['patient_id']
    reports = db.get_diagnosis_reports(patient_id)
    patient = db.get_patient(patient_id)
    
    # 患者专属统计
    today = datetime.date.today().strftime('%Y-%m-%d')
    pending_appointments = db.execute_query(
        """SELECT COUNT(*) as count FROM appointments 
           WHERE patient_id = %s AND status IN ('pending','confirmed')""",
        (patient_id,)
    )
    pending_appointments = pending_appointments[0]['count'] if pending_appointments else 0
    
    followups = db.get_followup_plans(patient_id) or []
    pending_followups = sum(1 for f in followups if f.get('status') == 'pending')
    
    unread_chat = db.get_total_unread_chat_count(current_user.id)
    unread_notifications = db.get_unread_notification_count(current_user.id)
    
    # 病灶趋势数据（用于图表）
    trend_data = db.get_patient_trend_data(patient_id) or []
    
    # 完成/待审核报告统计
    completed_reports = sum(1 for r in (reports or []) if r.get('status') == 'completed')
    pending_reports = sum(1 for r in (reports or []) if r.get('status') == 'pending')
    
    db.disconnect()
    
    return render_template('patient_dashboard.html', 
                         reports=reports or [], 
                         patient=patient[0] if patient else None,
                         pending_appointments=pending_appointments,
                         pending_followups=pending_followups,
                         unread_chat=unread_chat,
                         unread_notifications=unread_notifications,
                         trend_data=trend_data,
                         completed_reports=completed_reports,
                         pending_reports=pending_reports,
                         today=today,
                         today_str=datetime.datetime.now().strftime('%m月%d日'))

@app.route('/patient/trend')
@login_required
def patient_trend():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    patient = None
    if db.connect():
        rows = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
        patient = rows[0] if rows else None
        db.disconnect()
    return render_template('patient_trend.html', patient=patient)

@app.route('/patient/followup')
@login_required
def patient_followup():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    return render_template('patient_followup.html')



# 首页
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.user_type == 'admin':
            return redirect(url_for('admin_dashboard'))
        elif current_user.user_type == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))
    return redirect(url_for('login'))

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _disease_to_frontend(d):
    """将数据库 diseases 字段映射为前端模板/JS 期望的字段名"""
    if not d:
        return d
    d = dict(d)
    d.setdefault('icd_code', d.get('omim_id'))
    d.setdefault('diagnostic_criteria', d.get('diagnosis_methods'))
    d.setdefault('incidence', d.get('prevalence'))
    d.setdefault('prevention', d.get('references'))
    d.setdefault('inheritance', d.get('inheritance_pattern'))
    d.setdefault('is_featured', False)
    d.setdefault('category', '')
    return d

# 患者详情页
@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    db = Database()
    if not db.connect():
        return "数据库连接失败", 500
    
    patient = db.get_patient(patient_id)
    if not patient:
        return "患者不存在", 404
    
    images = db.get_medical_images(patient_id)
    reports = db.get_diagnosis_reports(patient_id)
    
    for report in reports:
        predictions = db.get_disease_predictions(report['report_id'])
        report['predictions'] = predictions
    
    # 患者统计与扩展信息
    followups = db.get_followup_plans(patient_id) or []
    trend_data = db.get_patient_trend_data(patient_id) or []
    appointments = db.execute_query(
        """SELECT a.*, u.full_name as doctor_name FROM appointments a
           JOIN users u ON a.doctor_id = u.user_id
           WHERE a.patient_id = %s ORDER BY a.appointment_date DESC, a.appointment_time DESC""",
        (patient_id,)
    ) or []
    
    # 该患者关联的 user_id（用于在线咨询）
    patient_user_id = patient[0].get('user_id')
    
    db.disconnect()
    model_trained = os.path.exists(MODEL_PATH)
    return render_template('patient.html', 
                          patient=patient[0], 
                          images=images, 
                          reports=reports,
                          followups=followups,
                          trend_data=trend_data,
                          appointments=appointments,
                          patient_user_id=patient_user_id,
                          total_images=len(images),
                          total_reports=len(reports),
                          total_followups=len(followups),
                          completed_reports=sum(1 for r in reports if r.get('status') == 'completed'),
                          pending_reports=sum(1 for r in reports if r.get('status') == 'pending'),
                          model_trained=model_trained)

# 添加患者
@app.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    contact_number = request.form.get('contact_number')
    medical_history = request.form.get('medical_history')
    
    if not name:
        return jsonify({'success': False, 'message': '姓名不能为空'})
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    patient_id = db.add_patient(name, age, gender, contact_number, medical_history)
    db.disconnect()
    
    if patient_id:
        return jsonify({'success': True, 'message': '患者添加成功', 'patient_id': patient_id})
    else:
        return jsonify({'success': False, 'message': '添加患者失败'})

# 上传影像（用于原患者详情页）
@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件部分'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        patient_id = request.form.get('patient_id')
        image_type = request.form.get('image_type')
        description = request.form.get('description')
        
        db = Database()
        if not db.connect():
            return jsonify({'success': False, 'message': '数据库连接失败'})
        
        image_id = db.add_medical_image(patient_id, unique_filename, image_type, description)
        db.disconnect()
        
        if image_id:
            return jsonify({'success': True, 'message': '文件上传成功', 'image_id': image_id})
        else:
            return jsonify({'success': False, 'message': '保存到数据库失败'})
    
    return jsonify({'success': False, 'message': '文件类型不允许'})

# 诊断接口（使用肺纤维化模型）
@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    patient_id = request.form.get('patient_id')
    clinical_notes = request.form.get('clinical_notes')
    
    if not patient_id:
        return jsonify({'success': False, 'message': '患者ID不能为空'})
    
    # 调用诊断服务（返回预测结果与基于模型输出的量化指标）
    result = pf_service.diagnose_patient(patient_id)
    if isinstance(result, dict):
        predictions = result['predictions']
        lesion_area_ratio = result['lesion_area_ratio']
        distribution_range = result['distribution_range']
        findings = result['imaging_findings']
        suggestions = result['suggestions']
        diagnosis_view = result.get('diagnosis_view') or {}
    else:
        predictions = result
        lesion_area_ratio = 0.0
        distribution_range = ''
        findings = ''
        suggestions = ''
        diagnosis_view = {}
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    doctor_id = current_user.id
    report_id = db.add_diagnosis_report(
        patient_id, doctor_id, clinical_notes, findings or "AI辅助诊断结果",
        lesion_area_ratio=lesion_area_ratio,
        distribution_range=distribution_range,
        suggestions='\n'.join(diagnosis_view.get('suggestions') or []),
        differentials=json.dumps(diagnosis_view.get('differentials') or [], ensure_ascii=False),
        conclusion_text=diagnosis_view.get('conclusion_text')
    )
    
    if report_id:
        for pred in predictions:
            db.add_disease_prediction(report_id, pred['disease_id'], pred['confidence'], pred['rank'])
        db.disconnect()
        return jsonify({'success': True, 'message': '诊断完成', 'report_id': report_id})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '创建诊断报告失败'})


# ---------- 医生复核（诊断报告与 AI 预测确认） ----------
@app.route('/doctor/reports')
@login_required
def doctor_reports():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('doctor_reports.html', reports=[], counts={}, diseases=[])
    reports = db.get_all_reports() or []
    for r in reports:
        if isinstance(r.get('created_at'), datetime.datetime):
            r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M')
        if isinstance(r.get('reviewed_at'), datetime.datetime):
            r['reviewed_at'] = r['reviewed_at'].strftime('%Y-%m-%d %H:%M')
        if isinstance(r.get('lesion_area_ratio'), Decimal):
            r['lesion_area_ratio'] = float(r['lesion_area_ratio'])
    counts = {
        'total': len(reports),
        'pending': sum(1 for r in reports if r.get('status') == 'pending'),
        'completed': sum(1 for r in reports if r.get('status') == 'completed'),
        'reviewed': sum(1 for r in reports if r.get('status') == 'reviewed'),
    }
    diseases = db.get_diseases() or []
    db.disconnect()
    return render_template('doctor_reports.html', reports=reports, counts=counts, diseases=diseases)


@app.route('/api/reports/<int:report_id>')
@login_required
def api_report_detail(report_id):
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'}), 500
    report = db.get_report_by_id(report_id)
    if not report:
        db.disconnect()
        return jsonify({'success': False, 'message': '报告不存在'}), 404
    report = report[0]
    predictions = db.get_disease_predictions(report_id) or []
    db.disconnect()
    # JSON 序列化兼容处理
    for p in predictions:
        p['confidence'] = float(p.get('confidence') or 0)
        p['is_confirmed'] = bool(p.get('is_confirmed'))
    for f in ('created_at', 'reviewed_at'):
        if isinstance(report.get(f), (datetime.datetime, datetime.date)):
            report[f] = report[f].strftime('%Y-%m-%d %H:%M')
    if isinstance(report.get('lesion_area_ratio'), Decimal):
        report['lesion_area_ratio'] = float(report['lesion_area_ratio'])
    return jsonify({'success': True, 'report': report, 'predictions': predictions})


@app.route('/api/reports/<int:report_id>/review', methods=['POST'])
@login_required
def api_report_review(report_id):
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    data = request.get_json(silent=True) or {}
    conclusion = (data.get('conclusion') or '').strip()
    status = data.get('status')
    predictions = data.get('predictions') or []
    # 随访联动：医生在复核时可填写建议随访日期，一键创建随访计划
    suggested_date = data.get('suggested_date') or None
    followup_notes = (data.get('followup_notes') or '').strip()
    if status not in ('completed', 'reviewed'):
        return jsonify({'success': False, 'message': '无效的复核状态'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    report = db.get_report_by_id(report_id)
    if not report:
        db.disconnect()
        return jsonify({'success': False, 'message': '报告不存在'})
    result = db.update_report_review(report_id, conclusion or '无', status)
    if result is None:
        db.disconnect()
        return jsonify({'success': False, 'message': '报告状态更新失败'})
    # 逐条写入 AI 预测的复核结果
    for p in predictions:
        pid = p.get('prediction_id')
        if not pid:
            continue
        is_confirmed = bool(p.get('is_confirmed'))
        notes = (p.get('notes') or '').strip()
        db.update_prediction_review(pid, is_confirmed, current_user.id if is_confirmed else None, notes or None)
    db.add_system_log(current_user.id, 'REVIEW_REPORT', f'医生复核报告，报告ID: {report_id}，状态: {status}')

    patient_id = report[0]['patient_id']
    followup_created = False
    # 联动创建随访计划
    if suggested_date:
        try:
            datetime.datetime.strptime(suggested_date, '%Y-%m-%d')
            plan_reason = followup_notes or f'基于诊断报告 #{report_id} 的复查建议'
            followup_created = bool(db.create_followup_plan(patient_id, suggested_date, plan_reason))
        except (ValueError, TypeError):
            followup_created = False

    # 通知患者
    patient = db.execute_query(
        "SELECT user_id FROM patients WHERE patient_id = %s", (patient_id,)
    )
    if patient and patient[0].get('user_id'):
        db.add_notification(
            patient[0]['user_id'],
            '诊断报告已更新',
            f'医生已完成您的诊断报告复核（编号 #{report_id}），请查看诊断结论。',
            'report',
            '/my_reports'
        )
        if followup_created:
            db.add_notification(
                patient[0]['user_id'],
                '新的随访计划',
                f'医生已为您安排随访复查，建议日期：{suggested_date}，请及时查看。',
                'followup',
                '/patient/followup'
            )
    db.disconnect()
    return jsonify({
        'success': True,
        'message': '复核已保存' + ('，并已创建随访计划' if followup_created else ''),
        'followup_created': followup_created
    })

# 模型训练接口
@app.route('/train_model', methods=['POST'])
@login_required
def train_model_route():
    return jsonify({'success': False, 'message': '模型已预置，无需在线训练'})

# 静态文件访问
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API: 获取所有患者
@app.route('/api/patients')
@login_required
def api_patients():
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    patients = db.get_patients()
    db.disconnect()
    return jsonify(patients)

# 个人资料重定向
@app.route('/profile')
@login_required
def profile():
    if current_user.user_type == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.user_type == 'doctor':
        return redirect(url_for('doctor_profile'))
    else:
        return redirect(url_for('patient_profile'))

# 医生个人资料
@app.route('/doctor_profile')
@login_required
def doctor_profile():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_profile'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('profile.html', user=None, patients_count=0, reports_count=0)
    
    user_data = db.execute_query("SELECT * FROM users WHERE user_id = %s", (current_user.id,))
    patients_data = db.execute_query("SELECT COUNT(*) as count FROM patients")
    patients_count = patients_data[0]['count'] if patients_data else 0
    reports_data = db.execute_query("SELECT COUNT(*) as count FROM diagnosis_reports")
    reports_count = reports_data[0]['count'] if reports_data else 0
    db.disconnect()
    
    if not user_data:
        flash('用户信息不存在', 'danger')
        return redirect(url_for('index'))
    
    return render_template('profile.html', 
                         user=user_data[0],
                         patients_count=patients_count,
                         reports_count=reports_count)

#更新患者信息
@app.route('/update_patient/<int:patient_id>', methods=['POST'])
@login_required
def update_patient(patient_id):
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    contact_number = request.form.get('contact_number')
    medical_history = request.form.get('medical_history')
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    result = db.execute_insert(
        """UPDATE patients SET name=%s, age=%s, gender=%s, contact_number=%s, medical_history=%s WHERE patient_id=%s""",
        (name, age, gender, contact_number, medical_history, patient_id)
    )
    db.disconnect()
    if result is not None:
        return jsonify({'success': True, 'message': '更新成功'})
    else:
        return jsonify({'success': False, 'message': '更新失败'})


# 患者个人档案
@app.route('/patient_profile')
@login_required
def patient_profile():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_profile.html', patient=None, reports_count=0)
    
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    reports_count = 0
    bmi = None
    follow_up_count = 0
    appointments_count = 0
    images_count = 0
    if patient_data:
        patient_id = patient_data[0]['patient_id']
        reports = db.get_diagnosis_reports(patient_id)
        reports_count = len(reports) if reports else 0
        follow_ups = db.get_followup_plans(patient_id)
        follow_up_count = len(follow_ups) if follow_ups else 0
        appt = db.execute_query(
            "SELECT COUNT(*) AS c FROM appointments WHERE patient_id = %s", (patient_id,))
        appointments_count = appt[0]['c'] if appt else 0
        imgs = db.execute_query(
            "SELECT COUNT(*) AS c FROM medical_images WHERE patient_id = %s", (patient_id,))
        images_count = imgs[0]['c'] if imgs else 0
        h = patient_data[0].get('height_cm')
        w = patient_data[0].get('weight_kg')
        if h and w:
            try:
                hm = float(h) / 100.0
                bmi = round(float(w) / (hm * hm), 1)
            except (TypeError, ValueError, ZeroDivisionError):
                bmi = None
    db.disconnect()

    return render_template('patient_profile.html',
                         patient=patient_data[0] if patient_data else None,
                         user=current_user,
                         reports_count=reports_count,
                         appointments_count=appointments_count,
                         images_count=images_count,
                         follow_up_count=follow_up_count,
                         bmi=bmi)

# 删除患者（软删除：标记 is_deleted，可恢复；彻底删除请使用回收站）
@app.route('/delete_patient/<int:patient_id>', methods=['DELETE'])
@login_required
def delete_patient(patient_id):
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    try:
        # 获取患者名称用于日志
        patient_data = db.get_patient(patient_id, include_deleted=True)
        patient_name = patient_data[0]['name'] if patient_data else '未知患者'
        if not patient_data:
            db.disconnect()
            return jsonify({'success': False, 'message': '患者不存在'})
        if patient_data[0].get('is_deleted'):
            db.disconnect()
            return jsonify({'success': False, 'message': '该患者已在回收站中'})

        # 软删除：仅标记 is_deleted=1，关联数据（诊断报告/影像/随访/预约/健康日志）全部保留可恢复
        result = db.soft_delete_patient(patient_id)

        if result:
            db.add_system_log(current_user.id, 'DELETE_PATIENT', f'软删除患者: {patient_name}（可恢复）')
            db.disconnect()
            return jsonify({'success': True, 'message': '患者已移入回收站，可随时恢复'})
        else:
            db.disconnect()
            return jsonify({'success': False, 'message': '删除患者失败'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'删除异常: {str(e)}'})


# API: 获取回收站中的已删除患者
@app.route('/api/patients/deleted')
@login_required
def api_deleted_patients():
    if current_user.user_type != 'admin':
        return jsonify({'error': '权限不足'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    patients = db.get_deleted_patients()
    # 附加每个已删除患者的关联数据统计（用于提示彻底删除影响）
    for p in patients:
        p['_stats'] = db.execute_query(
            """SELECT
                 (SELECT COUNT(*) FROM diagnosis_reports WHERE patient_id = %s) AS reports,
                 (SELECT COUNT(*) FROM medical_images WHERE patient_id = %s) AS images,
                 (SELECT COUNT(*) FROM followup_plans WHERE patient_id = %s) AS followups,
                 (SELECT COUNT(*) FROM appointments WHERE patient_id = %s) AS appointments,
                 (SELECT COUNT(*) FROM health_logs WHERE patient_id = %s) AS health_logs""",
            (p['patient_id'], p['patient_id'], p['patient_id'], p['patient_id'], p['patient_id'])
        )
        if p.get('_stats'):
            p['_stats'] = p['_stats'][0]
    db.disconnect()
    return jsonify(patients)


# API: 恢复已删除的患者
@app.route('/api/patients/<int:patient_id>/restore', methods=['POST'])
@login_required
def api_restore_patient(patient_id):
    if current_user.user_type != 'admin':
        return jsonify({'success': False, 'message': '无权操作'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    patient = db.get_patient(patient_id, include_deleted=True)
    if not patient:
        db.disconnect()
        return jsonify({'success': False, 'message': '患者不存在'})
    result = db.restore_patient(patient_id)
    if result:
        db.add_system_log(current_user.id, 'RESTORE_PATIENT', f'恢复患者: {patient[0]["name"]}')
    db.disconnect()
    return jsonify({'success': result is not None, 'message': '患者已恢复' if result else '恢复失败'})


# API: 彻底删除患者（不可恢复，依赖外键级联清理全部关联数据）
@app.route('/api/patients/<int:patient_id>/purge', methods=['DELETE'])
@login_required
def api_purge_patient(patient_id):
    if current_user.user_type != 'admin':
        return jsonify({'success': False, 'message': '无权操作'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    try:
        patient = db.get_patient(patient_id, include_deleted=True)
        if not patient:
            db.disconnect()
            return jsonify({'success': False, 'message': '患者不存在'})
        patient_name = patient[0]['name']
        # 只删除 patients 主记录，外键 ON DELETE CASCADE 自动清理
        # diagnosis_reports / medical_images / followup_plans / appointments / health_logs
        result = db.purge_patient(patient_id)
        if result:
            db.add_system_log(current_user.id, 'PURGE_PATIENT', f'彻底删除患者: {patient_name}（含全部关联数据）')
        db.disconnect()
        return jsonify({'success': bool(result), 'message': '患者已彻底删除' if result else '删除失败'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'彻底删除异常: {str(e)}'})

# API: 获取所有疾病（用于疾病查询页面）
@app.route('/api/diseases')
@login_required
def api_diseases():
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    diseases = db.get_diseases()
    db.disconnect()
    return jsonify([_disease_to_frontend(d) for d in (diseases or [])])

# API: 获取单个疾病详情
@app.route('/api/diseases/<int:disease_id>', methods=['GET'])
@login_required
def api_disease_detail(disease_id):
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    disease = db.execute_query("SELECT * FROM diseases WHERE disease_id = %s", (disease_id,))
    db.disconnect()
    if not disease:
        return jsonify({'error': '疾病不存在'}), 404
    return jsonify(_disease_to_frontend(disease[0]))

# API: 创建新疾病（仅医生）
@app.route('/api/diseases', methods=['POST'])
@login_required
def api_create_disease():
    if current_user.user_type != 'admin':
        return jsonify({'success': False, 'message': '权限不足'}), 403
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'message': '疾病名称不能为空'}), 400
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'}), 500
    disease_id = db.execute_insert(
        """INSERT INTO diseases (name, omim_id, description, symptoms, diagnosis_methods, 
           treatment_options, prevalence, inheritance_pattern, references) 
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (name, data.get('icd_code'), data.get('description'), data.get('symptoms'),
         data.get('diagnostic_criteria'), data.get('treatment_options'),
         data.get('incidence'), data.get('inheritance_pattern'),
         data.get('prevention'))
    )
    db.disconnect()
    if disease_id:
        return jsonify({'success': True, 'disease_id': disease_id})
    else:
        return jsonify({'success': False, 'message': '创建失败'}), 500

# API: 更新疾病（仅医生）
@app.route('/api/diseases/<int:disease_id>', methods=['PUT'])
@login_required
def api_update_disease(disease_id):
    if current_user.user_type != 'admin':
        return jsonify({'success': False, 'message': '权限不足'}), 403
    data = request.json
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'}), 500
    result = db.execute_insert(
        """UPDATE diseases SET name=%s, omim_id=%s, description=%s, symptoms=%s, 
           diagnosis_methods=%s, treatment_options=%s, prevalence=%s, inheritance_pattern=%s, 
           references=%s 
           WHERE disease_id=%s""",
        (data.get('name'), data.get('icd_code'), data.get('description'), data.get('symptoms'),
         data.get('diagnostic_criteria'), data.get('treatment_options'),
         data.get('incidence'), data.get('inheritance_pattern'),
         data.get('prevention'), disease_id)
    )
    db.disconnect()
    if result is not None:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': '更新失败'}), 500

# API: 删除疾病（仅医生）
@app.route('/api/diseases/<int:disease_id>', methods=['DELETE'])
@login_required
def api_delete_disease(disease_id):
    if current_user.user_type != 'admin':
        return jsonify({'success': False, 'message': '权限不足'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'}), 500
    # 先删除关联的预测记录（防止外键约束）
    db.execute_insert("DELETE FROM disease_predictions WHERE disease_id=%s", (disease_id,))
    result = db.execute_insert("DELETE FROM diseases WHERE disease_id=%s", (disease_id,))
    db.disconnect()
    if result:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'message': '删除失败'}), 500

# 疾病管理页面
@app.route('/disease_management', methods=['GET', 'POST'])
@login_required
def disease_management():
    if current_user.user_type != 'admin':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('disease_management.html', diseases=[])
    if request.method == 'POST':
        action = request.form.get('action', 'add')
        name = request.form.get('name')
        description = request.form.get('description')
        symptoms = request.form.get('symptoms')
        treatment = request.form.get('treatment')
        if action == 'add' and name:
            try:
                disease_id = db.add_disease(name, description, symptoms, treatment)
                if disease_id:
                    db.add_system_log(current_user.id, 'ADD_DISEASE', f'添加疾病: {name}')
                    db.disconnect()
                    return jsonify({'success': True, 'message': '疾病添加成功'})
                else:
                    db.disconnect()
                    return jsonify({'success': False, 'message': '添加疾病失败'})
            except Exception as e:
                db.disconnect()
                return jsonify({'success': False, 'message': f'添加异常: {str(e)}'})
        else:
            db.disconnect()
            return jsonify({'success': False, 'message': '疾病名称不能为空'})
    diseases = db.get_diseases()
    db.disconnect()
    return render_template('disease_management.html', diseases=diseases)

# 系统日志页面
@app.route('/system_logs')
@login_required
def system_logs():
    if current_user.user_type != 'admin':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('system_logs.html', logs=[])
    logs = db.get_system_logs()
    db.disconnect()
    # 如果查询返回 None，则设为空列表
    if logs is None:
        logs = []
    return render_template('system_logs.html', logs=logs)


# ==================== 管理员端 ====================
@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    db = Database()
    if not db.connect():
        return render_template('admin_dashboard.html', stats={}, recent_logs=[])

    def count(sql, args=None):
        r = db.execute_query(sql, args or ())
        return r[0]['c'] if r else 0

    stats = {
        'users': count("SELECT COUNT(*) AS c FROM users"),
        'doctors': count("SELECT COUNT(*) AS c FROM users WHERE role='doctor'"),
        'patients': count("SELECT COUNT(*) AS c FROM users WHERE role='patient'"),
        'admins': count("SELECT COUNT(*) AS c FROM users WHERE role='admin'"),
        'patient_records': count("SELECT COUNT(*) AS c FROM patients WHERE is_deleted=0"),
        'reports': count("SELECT COUNT(*) AS c FROM diagnosis_reports"),
        'images': count("SELECT COUNT(*) AS c FROM medical_images"),
        'diseases': count("SELECT COUNT(*) AS c FROM diseases"),
        'logs': count("SELECT COUNT(*) AS c FROM system_logs"),
    }
    recent_logs = db.get_system_logs(limit=10) or []
    db.disconnect()
    return render_template('admin_dashboard.html', stats=stats, recent_logs=recent_logs)


@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    db = Database()
    if not db.connect():
        return render_template('admin_users.html', users=[])
    users = db.execute_query(
        """SELECT u.*, (SELECT p.name FROM patients p WHERE p.user_id = u.user_id LIMIT 1) AS patient_name
           FROM users u ORDER BY u.user_id""") or []
    db.disconnect()
    return render_template('admin_users.html', users=users)


@app.route('/admin/users/create', methods=['POST'])
@login_required
@admin_required
def admin_create_user():
    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''
    full_name = (request.form.get('full_name') or '').strip()
    email = (request.form.get('email') or '').strip()
    role = request.form.get('role')
    if not username or not password or not full_name:
        return jsonify({'success': False, 'message': '用户名、密码、姓名不能为空'})
    if role not in ('doctor', 'patient'):
        return jsonify({'success': False, 'message': '角色无效'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    if db.execute_query("SELECT user_id FROM users WHERE username=%s", (username,)):
        db.disconnect()
        return jsonify({'success': False, 'message': '用户名已存在'})
    user_id = db.execute_insert(
        "INSERT INTO users (username, password_hash, email, full_name, role) VALUES (%s,%s,%s,%s,%s)",
        (username, generate_password_hash(password), email, full_name, role))
    if role == 'patient' and user_id:
        db.execute_insert("INSERT INTO patients (name, user_id) VALUES (%s,%s)", (full_name, user_id))
    db.add_system_log(current_user.id, 'CREATE_USER', f'创建账号: {username}（{role}）')
    db.disconnect()
    return jsonify({'success': bool(user_id), 'message': '账号创建成功' if user_id else '创建失败'})


@app.route('/admin/users/<int:user_id>/reset_password', methods=['POST'])
@login_required
@admin_required
def admin_reset_password(user_id):
    new_pwd = request.form.get('password') or ''
    if len(new_pwd) < 6:
        return jsonify({'success': False, 'message': '密码至少6位'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    result = db.execute_insert(
        "UPDATE users SET password_hash=%s WHERE user_id=%s",
        (generate_password_hash(new_pwd), user_id))
    if result is not None:
        db.add_system_log(current_user.id, 'RESET_PASSWORD', f'重置账号密码: user_id={user_id}')
    db.disconnect()
    return jsonify({'success': result is not None, 'message': '密码已重置' if result is not None else '重置失败'})


@app.route('/admin/recycle_bin')
@login_required
@admin_required
def admin_recycle_bin():
    return render_template('admin_recycle_bin.html')


# API: 添加系统日志
@app.route('/api/logs', methods=['POST'])
@login_required
def api_logs():
    try:
        data = request.json
        user_id = current_user.id
        action = data.get('action')
        details = data.get('details', '')
        db = Database()
        if not db.connect():
            return jsonify({'success': False, 'message': '数据库连接失败'}), 500
        log_id = db.add_system_log(user_id, action, details)
        db.disconnect()
        return jsonify({'success': True, 'log_id': log_id})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# 症状自查（模拟）
@app.route('/symptom_check', methods=['GET', 'POST'])
@login_required
def symptom_check():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        symptom_details = request.form.get('symptomDetails')
        age = request.form.get('age')
        gender = request.form.get('gender')
        medical_history = request.form.get('medicalHistory')
        possible_diseases = analyze_symptoms(symptoms, symptom_details, age, gender, medical_history)
        return render_template('symptom_result.html', 
                             possible_diseases=possible_diseases,
                             user_input={
                                 'symptoms': symptoms,
                                 'symptom_details': symptom_details,
                                 'age': age,
                                 'gender': gender,
                                 'medical_history': medical_history
                             })
    return render_template('symptom_check.html')

def analyze_symptoms(symptoms, details, age, gender, history):
    # 返回模拟数据，可根据需要修改
    return [
        {
            'name': '特发性肺纤维化(IPF)',
            'match_score': 0.85,
            'description': '最常见的特发性间质性肺炎，呈进行性肺纤维化',
            'common_symptoms': ['干咳', '活动后呼吸困难', '杵状指'],
            'icd_code': 'J84.1'
        },
        {
            'name': '过敏性肺炎(HP)',
            'match_score': 0.60,
            'description': '由吸入抗原引起的免疫介导性肺病',
            'common_symptoms': ['发热', '咳嗽', '呼吸困难'],
            'icd_code': 'J67.9'
        }
    ]

# 删除疾病（旧路由，保留兼容）
@app.route('/delete_disease/<int:disease_id>', methods=['DELETE'])
@login_required
def delete_disease(disease_id):
    if current_user.user_type != 'admin':
        return jsonify({'success': False, 'message': '无权操作'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    try:
        disease_data = db.execute_query("SELECT name FROM diseases WHERE disease_id = %s", (disease_id,))
        disease_name = disease_data[0]['name'] if disease_data else '未知疾病'
        result = db.execute_insert("DELETE FROM diseases WHERE disease_id = %s", (disease_id,))
        if result:
            db.add_system_log(current_user.id, 'DELETE_DISEASE', f'删除疾病: {disease_name}')
            db.disconnect()
            return jsonify({'success': True, 'message': '疾病删除成功'})
        else:
            db.disconnect()
            return jsonify({'success': False, 'message': '删除疾病失败'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'删除异常: {str(e)}'})

# 患者列表
@app.route('/patient_list')
@login_required
def patient_list():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_list.html', patients=[])
    patients = db.get_patients()
    db.disconnect()
    return render_template('patient_list.html', patients=patients)

# 删除患者
# @app.route('/delete_patient/<int:patient_id>', methods=['DELETE'])
# @login_required
# def delete_patient(patient_id):
#     if current_user.user_type != 'doctor':
#         return jsonify({'success': False, 'message': '无权操作'})
#     db = Database()
#     if not db.connect():
#         return jsonify({'success': False, 'message': '数据库连接失败'})
#     try:
#         patient_data = db.execute_query("SELECT name FROM patients WHERE patient_id = %s", (patient_id,))
#         patient_name = patient_data[0]['name'] if patient_data else '未知患者'
#         result = db.execute_insert("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
#         if result:
#             db.add_system_log(current_user.id, 'DELETE_PATIENT', f'删除患者: {patient_name}')
#             db.disconnect()
#             return jsonify({'success': True, 'message': '患者删除成功'})
#         else:
#             db.disconnect()
#             return jsonify({'success': False, 'message': '删除患者失败'})
#     except Exception as e:
#         db.disconnect()
#         return jsonify({'success': False, 'message': f'删除异常: {str(e)}'})

# 医生今日日程（接入真实预约数据）
@app.route('/doctor/schedule')
@login_required
def doctor_schedule():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('doctor_schedule.html', appointments=[], patients=[], today='')
    # 该医生名下全部预约（关联患者信息）
    appointments = db.execute_query(
        """SELECT a.*, p.name AS patient_name, p.user_id AS patient_user_id
           FROM appointments a
           JOIN patients p ON a.patient_id = p.patient_id
           WHERE a.doctor_id = %s
           ORDER BY a.appointment_date DESC, a.appointment_time DESC""",
        (current_user.id,)
    ) or []
    # 将日期/时间字段转为字符串，避免 JSON 序列化问题（TIME 类型读回为 timedelta）
    for a in appointments:
        if hasattr(a.get('appointment_date'), 'strftime'):
            a['appointment_date'] = a['appointment_date'].strftime('%Y-%m-%d')
        if isinstance(a.get('appointment_time'), datetime.timedelta):
            total_seconds = int(a['appointment_time'].total_seconds())
            a['appointment_time'] = f'{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}'
        for f in ('created_at', 'updated_at'):
            if hasattr(a.get(f), 'strftime'):
                a[f] = a[f].strftime('%Y-%m-%d %H:%M:%S')
    patients = db.get_patients() or []
    db.disconnect()
    return render_template('doctor_schedule.html',
                           appointments=appointments,
                           patients=patients,
                           today=datetime.date.today().strftime('%Y-%m-%d'))


def _doctor_update_appointment(appointment_id, new_status, log_action, success_msg, notify_title, notify_msg_template):
    """医生端预约状态更新公共逻辑，返回 jsonify 响应"""
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    appointment = db.execute_query(
        "SELECT * FROM appointments WHERE appointment_id = %s AND doctor_id = %s",
        (appointment_id, current_user.id)
    )
    if not appointment:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到该预约记录'})
    appt = appointment[0]
    if appt.get('status') == 'cancelled':
        db.disconnect()
        return jsonify({'success': False, 'message': '该预约已取消，无法操作'})
    result = db.execute_update(
        "UPDATE appointments SET status = %s, updated_at = %s WHERE appointment_id = %s",
        (new_status, datetime.datetime.now(), appointment_id)
    )
    if result is not None and result > 0:
        db.add_system_log(current_user.id, log_action, f'{log_action}，预约ID: {appointment_id}')
        # 通知患者（格式化预约时间，TIME 类型读回为 timedelta）
        notify_date = appt.get('appointment_date')
        notify_time = appt.get('appointment_time')
        if isinstance(notify_date, datetime.date) and not isinstance(notify_date, datetime.datetime):
            notify_date = notify_date.strftime('%Y-%m-%d')
        if isinstance(notify_time, datetime.timedelta):
            total_seconds = int(notify_time.total_seconds())
            notify_time = f'{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}'
        patient = db.execute_query(
            "SELECT user_id FROM patients WHERE patient_id = %s", (appt['patient_id'],)
        )
        if patient and patient[0].get('user_id'):
            db.add_notification(
                patient[0]['user_id'],
                notify_title,
                notify_msg_template.format(date=notify_date, time=notify_time),
                'appointment',
                '/patient/appointment'
            )
        db.disconnect()
        return jsonify({'success': True, 'message': success_msg})
    db.disconnect()
    return jsonify({'success': False, 'message': '操作失败'})


# 医生确认预约
@app.route('/doctor/appointment/<int:appointment_id>/confirm', methods=['POST'])
@login_required
def doctor_confirm_appointment(appointment_id):
    return _doctor_update_appointment(
        appointment_id, 'confirmed', 'CONFIRM_APPOINTMENT', '预约已确认',
        '预约已确认', '您 {date} {time} 的预约已由医生确认，请按时就诊。'
    )


# 医生完成预约（就诊结束）
@app.route('/doctor/appointment/<int:appointment_id>/complete', methods=['POST'])
@login_required
def doctor_complete_appointment(appointment_id):
    return _doctor_update_appointment(
        appointment_id, 'completed', 'COMPLETE_APPOINTMENT', '预约已完成',
        '就诊已完成', '您 {date} {time} 的就诊已完成，祝您早日康复。'
    )


# 医生取消预约
@app.route('/doctor/appointment/<int:appointment_id>/cancel', methods=['POST'])
@login_required
def doctor_cancel_appointment(appointment_id):
    return _doctor_update_appointment(
        appointment_id, 'cancelled', 'CANCEL_APPOINTMENT', '预约已取消',
        '预约已取消', '很抱歉，您 {date} {time} 的预约已被医生取消，如有疑问请联系医生。'
    )


# 医生为患者新建预约
@app.route('/doctor/create_appointment', methods=['POST'])
@login_required
def doctor_create_appointment():
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    patient_id = request.form.get('patient_id')
    appointment_date = request.form.get('appointment_date')
    appointment_time = request.form.get('appointment_time')
    department = request.form.get('department') or '呼吸科'
    symptoms = request.form.get('symptoms') or ''
    notes = request.form.get('notes') or ''
    if not all([patient_id, appointment_date, appointment_time]):
        return jsonify({'success': False, 'message': '请填写完整的预约信息'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    patient = db.execute_query("SELECT user_id, name FROM patients WHERE patient_id = %s", (patient_id,))
    if not patient:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到患者信息'})
    # 检查时间段冲突
    existing = db.execute_query(
        """SELECT * FROM appointments
           WHERE doctor_id = %s AND appointment_date = %s AND appointment_time = %s AND status != 'cancelled'""",
        (current_user.id, appointment_date, appointment_time)
    )
    if existing:
        db.disconnect()
        return jsonify({'success': False, 'message': '该时间段已有预约，请选择其他时间'})
    appointment_id = db.execute_insert(
        """INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time,
           department, symptoms, notes, status, created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, 'confirmed', %s)""",
        (patient_id, current_user.id, appointment_date, appointment_time, department, symptoms, notes, datetime.datetime.now())
    )
    if appointment_id:
        db.add_system_log(current_user.id, 'CREATE_APPOINTMENT', f'医生新建预约，预约ID: {appointment_id}')
        if patient[0].get('user_id'):
            db.add_notification(
                patient[0]['user_id'],
                '医生为您安排了预约',
                f'医生为您预约了 {appointment_date} {appointment_time} 的门诊，请按时就诊。',
                'appointment',
                '/patient/appointment'
            )
        db.disconnect()
        return jsonify({'success': True, 'message': '预约创建成功', 'appointment_id': appointment_id})
    db.disconnect()
    return jsonify({'success': False, 'message': '预约创建失败'})

# 疾病查询页面（肺纤维化主题）
@app.route('/disease_query')
@login_required
def disease_query():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('disease_query.html', diseases=[], featured_diseases=[])
    diseases = [_disease_to_frontend(d) for d in (db.get_diseases() or [])]
    featured_diseases = [_disease_to_frontend(d) for d in (db.execute_query(
        "SELECT * FROM diseases ORDER BY disease_id DESC LIMIT 6"
    ) or [])]
    db.disconnect()
    return render_template('disease_query.html', 
                         diseases=diseases, 
                         featured_diseases=featured_diseases)

# 患者预约挂号
@app.route('/patient/appointment')
@login_required
def patient_appointment():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_appointment.html', doctors=[], appointments=[])
    doctors = db.execute_query(
        """SELECT user_id, full_name, specialization, hospital_affiliation, license_number
           FROM users WHERE role = 'doctor'"""
    ) or []
    # 兼容旧模板字段名：specialty -> specialization, department -> hospital_affiliation, title -> 默认职称
    for d in doctors:
        d['specialty'] = d.get('specialization') or '间质性肺病诊疗'
        d['department'] = d.get('hospital_affiliation') or '呼吸科'
        d['title'] = '医师'
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    appointments = []
    if patient_data:
        patient_id = patient_data[0]['patient_id']
        appointments = db.execute_query(
            """SELECT a.*, u.full_name as doctor_name, u.specialization as specialty, 
                      u.hospital_affiliation as department, u.license_number
               FROM appointments a 
               JOIN users u ON a.doctor_id = u.user_id 
               WHERE a.patient_id = %s 
               ORDER BY a.appointment_date DESC, a.appointment_time DESC""",
            (patient_id,)
        ) or []
        # 格式化日期/时间（TIME 类型读回为 timedelta）
        for a in appointments:
            if hasattr(a.get('appointment_date'), 'strftime'):
                a['appointment_date'] = a['appointment_date'].strftime('%Y-%m-%d')
            if isinstance(a.get('appointment_time'), datetime.timedelta):
                total_seconds = int(a['appointment_time'].total_seconds())
                a['appointment_time'] = f'{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}'
    stats = {'pending': 0, 'confirmed': 0, 'completed': 0, 'cancelled': 0}
    for a in appointments:
        s = a.get('status')
        if s in stats:
            stats[s] += 1
    db.disconnect()
    return render_template('patient_appointment.html', doctors=doctors, appointments=appointments,
                           stats=stats, today=datetime.date.today().strftime('%Y-%m-%d'))


# API: 查询某医生某日已约时段（用于前端禁用已满时段）
@app.route('/api/appointments/slots')
@login_required
def api_appointment_slots():
    if current_user.user_type != 'patient':
        return jsonify({'error': '权限不足'}), 403
    doctor_id = request.args.get('doctor_id')
    date = request.args.get('date')
    if not doctor_id or not date:
        return jsonify({'error': '缺少参数'}), 400
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    rows = db.execute_query(
        """SELECT appointment_time FROM appointments
           WHERE doctor_id=%s AND appointment_date=%s AND status != 'cancelled'""",
        (doctor_id, date)) or []
    occupied = []
    for r in rows:
        t = r.get('appointment_time')
        if isinstance(t, datetime.timedelta):
            total_seconds = int(t.total_seconds())
            t = f'{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}'
        occupied.append(str(t))
    db.disconnect()
    return jsonify({'occupied': occupied})


# API: 预约详情（真实数据，替换原写死假数据）
@app.route('/api/appointments/<int:appointment_id>')
@login_required
def api_appointment_detail(appointment_id):
    if current_user.user_type != 'patient':
        return jsonify({'error': '权限不足'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    p = db.execute_query("SELECT patient_id FROM patients WHERE user_id=%s", (current_user.id,))
    if not p:
        db.disconnect()
        return jsonify({'error': '未找到患者信息'}), 404
    rows = db.execute_query(
        """SELECT a.*, u.full_name AS doctor_name, u.specialization, u.hospital_affiliation,
                  u.license_number
           FROM appointments a JOIN users u ON a.doctor_id=u.user_id
           WHERE a.appointment_id=%s AND a.patient_id=%s""",
        (appointment_id, p[0]['patient_id'])) or []
    if not rows:
        db.disconnect()
        return jsonify({'error': '预约不存在'}), 404
    a = rows[0]
    if hasattr(a.get('appointment_date'), 'strftime'):
        a['appointment_date'] = a['appointment_date'].strftime('%Y-%m-%d')
    if isinstance(a.get('appointment_time'), datetime.timedelta):
        total_seconds = int(a['appointment_time'].total_seconds())
        a['appointment_time'] = f'{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}'
    for f in ('created_at', 'updated_at'):
        if hasattr(a.get(f), 'strftime'):
            a[f] = a[f].strftime('%Y-%m-%d %H:%M')
    db.disconnect()
    return jsonify(a)

# 创建预约
@app.route('/make_appointment', methods=['POST'])
@login_required
def make_appointment():
    if current_user.user_type != 'patient':
        return jsonify({'success': False, 'message': '无权操作'})
    doctor_id = request.form.get('doctor_id')
    appointment_date = request.form.get('appointment_date')
    appointment_time = request.form.get('appointment_time')
    department = request.form.get('department')
    symptoms = request.form.get('symptoms')
    notes = request.form.get('notes')
    if not all([doctor_id, appointment_date, appointment_time]):
        return jsonify({'success': False, 'message': '请填写完整的预约信息'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到患者信息'})
    patient_id = patient_data[0]['patient_id']
    existing_appointment = db.execute_query(
        """SELECT * FROM appointments 
           WHERE doctor_id = %s AND appointment_date = %s AND appointment_time = %s AND status != 'cancelled'""",
        (doctor_id, appointment_date, appointment_time)
    )
    if existing_appointment:
        db.disconnect()
        return jsonify({'success': False, 'message': '该时间段已被预约，请选择其他时间'})
    appointment_id = db.execute_insert(
        """INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, 
           department, symptoms, notes, status, created_at) 
           VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s)""",
        (patient_id, doctor_id, appointment_date, appointment_time, department, symptoms, notes, datetime.datetime.now())
    )
    if appointment_id:
        db.add_system_log(current_user.id, 'CREATE_APPOINTMENT', f'患者创建预约，预约ID: {appointment_id}')
        # 发送消息通知：通知患者 + 通知医生
        db.add_notification(
            current_user.id,
            '预约提交成功',
            f'您已成功预约 {appointment_date} {appointment_time} 的门诊，医院工作人员将在24小时内与您确认。',
            'appointment',
            '/patient/appointment'
        )
        doctor_user = db.execute_query("SELECT user_id FROM users WHERE user_id = %s", (doctor_id,))
        if doctor_user:
            patient_name = patient_data[0].get('name') or current_user.full_name or '患者'
            db.add_notification(
                doctor_user[0]['user_id'],
                '新预约待确认',
                f'患者 {patient_name} 预约了 {appointment_date} {appointment_time} 的门诊，请及时确认。',
                'appointment',
                '/doctor/schedule'
            )
        db.disconnect()
        return jsonify({'success': True, 'message': '预约成功', 'appointment_id': appointment_id})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '预约失败'})

# 取消预约
@app.route('/cancel_appointment/<int:appointment_id>', methods=['POST'])
@login_required
def cancel_appointment(appointment_id):
    if current_user.user_type != 'patient':
        return jsonify({'success': False, 'message': '无权操作'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到患者信息'})
    patient_id = patient_data[0]['patient_id']
    appointment_data = db.execute_query(
        "SELECT * FROM appointments WHERE appointment_id = %s AND patient_id = %s",
        (appointment_id, patient_id)
    )
    if not appointment_data:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到预约记录'})
    if appointment_data[0].get('status') in ('cancelled', 'completed'):
        db.disconnect()
        return jsonify({'success': False, 'message': '该预约当前状态不可取消'})
    result = db.execute_update(
        "UPDATE appointments SET status = 'cancelled', updated_at = %s WHERE appointment_id = %s",
        (datetime.datetime.now(), appointment_id)
    )
    if result is not None and result > 0:
        db.add_system_log(current_user.id, 'CANCEL_APPOINTMENT', f'患者取消预约，预约ID: {appointment_id}')
        db.disconnect()
        return jsonify({'success': True, 'message': '预约已取消'})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '取消预约失败'})

# 患者在线咨询界面
@app.route('/patient/chat')
@login_required
def patient_chat():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_chat.html', patient=None, messages=[], online_doctors=[])
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    patient = patient_data[0] if patient_data else None
    online_doctors = db.execute_query(
        """SELECT user_id, full_name, specialization as specialty, hospital_affiliation as department 
           FROM users WHERE role = 'doctor' ORDER BY user_id DESC"""
    ) or []
    # 获取患者历史会话
    conversations = db.get_user_conversations(current_user.id, 'patient') or []
    for conv in conversations:
        conv['unread_count'] = db.get_conversation_unread_count(conv['conversation_id'], current_user.id)
    db.disconnect()
    return render_template('patient_chat.html', 
                         patient=patient,
                         messages=[],
                         online_doctors=online_doctors,
                         conversations=conversations)

# ==================== 新增 AI 诊断独立页面 ====================
@app.route('/doctor/ai_diagnosis')
@login_required
def doctor_ai_diagnosis():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    db = Database()
    if not db.connect():
        patients = []
    else:
        patients = db.get_patients()
        db.disconnect()
    return render_template('ai_diagnosis.html', patients=patients)

@app.route('/api/ai_diagnose', methods=['POST'])
@login_required
def api_ai_diagnose():
    try:
        patient_id = request.form.get('patient_id')
        if not patient_id:
            return jsonify({'success': False, 'message': '请选择患者'})
        files = request.files.getlist('images')
        if not files:
            return jsonify({'success': False, 'message': '请至少上传一张CT切片'})

        db = Database()
        if not db.connect():
            return jsonify({'success': False, 'message': '数据库连接失败'})

        saved_paths = []
        thumbnails = []      # 缩略图 URL 列表（150x150）
        previews = []        # 预览图 URL 列表（800宽）
        
        # 确保目录存在
        upload_dir = app.config['UPLOAD_FOLDER']
        thumb_dir = os.path.join('static', 'thumbnails')
        preview_dir = os.path.join('static', 'previews')
        os.makedirs(thumb_dir, exist_ok=True)
        os.makedirs(preview_dir, exist_ok=True)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(upload_dir, unique_filename)
                file.save(filepath)

                # 读取图像（DICOM 或普通图片）
                ext = os.path.splitext(unique_filename)[1].lower()
                if ext == '.dcm':
                    dcm = pydicom.dcmread(filepath)
                    img = dcm.pixel_array.astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img).convert('RGB')
                else:
                    img_pil = Image.open(filepath).convert('RGB')

                # 生成预览图（宽度 800，高度等比）
                preview_img = img_pil.copy()
                preview_img.thumbnail((800, 800))
                preview_filename = f"preview_{unique_filename}.png"
                preview_path = os.path.join(preview_dir, preview_filename)
                preview_img.save(preview_path)
                preview_url = f'/static/previews/{preview_filename}'
                previews.append(preview_url)

                # 生成缩略图（150x150）
                thumb_img = img_pil.copy()
                thumb_img.thumbnail((150, 150))
                thumb_filename = f"thumb_{unique_filename}.png"
                thumb_path = os.path.join(thumb_dir, thumb_filename)
                thumb_img.save(thumb_path)
                thumbnail_url = f'/static/thumbnails/{thumb_filename}'
                thumbnails.append(thumbnail_url)

                # 保存记录到数据库（存储原始文件名）
                db.add_medical_image(patient_id, unique_filename, 'CT', 'AI诊断上传')
                saved_paths.append(filepath)

        db.disconnect()
        if not saved_paths:
            return jsonify({'success': False, 'message': '文件上传失败'})

        # 调用诊断服务（返回预测、热力图、量化指标等），诊断耗时取真实计时
        t0 = time.time()
        predictions, heatmap_url, lesion_ratio, distribution, findings, suggestions, diagnosis_view = \
            pf_service.predict_from_paths(saved_paths, patient_id)
        time_cost = round(time.time() - t0, 1)

        # 保存诊断报告：患者端「我的报告/病灶趋势」、医生端「报告复核」均依赖此记录，
        # 避免诊断结果仅在页面停留、退出后消失
        report_id = None
        try:
            rdb = Database()
            if rdb.connect():
                disease_id = pf_service._resolve_catalog_disease_id()
                report_id = rdb.add_diagnosis_report(
                    patient_id, current_user.id, '',
                    findings or 'AI辅助诊断结果',
                    lesion_area_ratio=lesion_ratio,
                    distribution_range=distribution,
                    suggestions='\n'.join(diagnosis_view.get('suggestions') or []),
                    differentials=json.dumps(diagnosis_view.get('differentials') or [], ensure_ascii=False),
                    conclusion_text=diagnosis_view.get('conclusion_text'))
                if report_id and predictions:
                    for pred in predictions:
                        rdb.add_disease_prediction(
                            report_id, disease_id,
                            pred.get('confidence', 0), pred.get('rank', 1))
                rdb.disconnect()
        except Exception as e:
            print(f"保存诊断报告失败: {e}")

        # 通知患者：AI 诊断已完成
        try:
            ndb = Database()
            if ndb.connect():
                patient_row = ndb.execute_query("SELECT user_id FROM patients WHERE patient_id = %s", (patient_id,))
                if patient_row:
                    ndb.add_notification(
                        patient_row[0]['user_id'],
                        'AI诊断报告已生成',
                        f'您的AI辅助诊断已完成，可前往「我的报告」查看详细结果。',
                        'diagnosis',
                        '/my_reports'
                    )
                ndb.disconnect()
        except Exception:
            pass

        return jsonify({
            'success': True,
            'report_id': report_id,
            'predictions': predictions,
            'primary_diagnosis': diagnosis_view.get('primary_diagnosis'),
            'icd_code': diagnosis_view.get('icd_code'),
            'differentials': diagnosis_view.get('differentials', []),
            'conclusion_text': diagnosis_view.get('conclusion_text'),
            'imaging_blocks': diagnosis_view.get('imaging_blocks', []),
            'suggestions_list': diagnosis_view.get('suggestions', []),
            'heatmap_url': heatmap_url,
            'lesion_area_ratio': lesion_ratio,
            'distribution_range': distribution,
            'imaging_findings': findings,
            'suggestions': suggestions,
            'time_cost': time_cost,
            'n_slices': int(distribution.split('/')[0]) if '/' in str(distribution) else len(saved_paths),
            'model_version': os.path.basename(MODEL_PATH),
            'thumbnails': thumbnails,     # 缩略图列表（用于缩略图栏）
            'previews': previews          # 预览图列表（用于主图显示）
        })
    except Exception as e:
        print(f"AI诊断错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'AI诊断服务暂时不可用，请稍后重试'})


@app.route('/api/export_report', methods=['POST'])
@login_required
def api_export_report():
    """导出规范化的 AI 辅助诊断报告 PDF"""
    try:
        data = request.get_json(silent=True) or {}
        patient_id = data.get('patient_id')
        if not patient_id:
            return jsonify({'success': False, 'message': '缺少患者ID'}), 400

        patient_name = ''
        db = Database()
        if db.connect():
            row = db.execute_query(
                "SELECT name FROM patients WHERE patient_id = %s", (patient_id,))
            if row:
                patient_name = row[0].get('name', '')
            db.disconnect()

        from pdf_report import build_diagnosis_report_pdf
        pdf_bytes = build_diagnosis_report_pdf(
            data,
            doctor_name=current_user.full_name or current_user.username or '',
            patient_name=patient_name
        )
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        filename = f'肺影智诊_AI诊断报告_患者{patient_id}_{now}.pdf'
        return send_file(io.BytesIO(pdf_bytes), mimetype='application/pdf',
                         as_attachment=True, download_name=filename)
    except Exception as e:
        print(f"导出PDF失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': '报告导出失败，请稍后重试'})

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """更新当前用户的基本信息"""
    user_id = current_user.id
    full_name = request.form.get('full_name')
    email = request.form.get('email')
    # 可选：专业领域、个人简介等（需要 users 表有相应字段，若没有可先忽略或扩展）
    specialty = request.form.get('specialty')
    bio = request.form.get('bio')
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    # 更新 users 表
    update_query = "UPDATE users SET full_name=%s, email=%s WHERE user_id=%s"
    params = [full_name, email, user_id]
    db.execute_insert(update_query, params)
    
    # 如果是医生，可能还有 specialization 等字段，如果表中有则更新
    if current_user.user_type == 'doctor' and specialty:
        db.execute_insert("UPDATE users SET specialization=%s WHERE user_id=%s", (specialty, user_id))
    
    db.disconnect()
    return jsonify({'success': True, 'message': '资料更新成功'})

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    old_pwd = request.form.get('old_password')
    new_pwd = request.form.get('new_password')
    confirm = request.form.get('confirm_password')
    
    if new_pwd != confirm:
        return jsonify({'success': False, 'message': '两次新密码不一致'})
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    user_data = db.execute_query("SELECT password_hash FROM users WHERE user_id=%s", (current_user.id,))
    if not user_data or not check_password_hash(user_data[0]['password_hash'], old_pwd):
        db.disconnect()
        return jsonify({'success': False, 'message': '原密码错误'})
    
    new_hash = generate_password_hash(new_pwd)
    db.execute_insert("UPDATE users SET password_hash=%s WHERE user_id=%s", (new_hash, current_user.id))
    db.disconnect()
    return jsonify({'success': True, 'message': '密码修改成功，请重新登录'})

@app.route('/update_patient_profile', methods=['POST'])
@login_required
def update_patient_profile():
    if current_user.user_type != 'patient':
        return jsonify({'success': False, 'message': '无权操作'})

    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})

    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id=%s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到患者信息'})

    patient_id = patient_data[0]['patient_id']
    name = (request.form.get('name') or '').strip()
    if not name:
        db.disconnect()
        return jsonify({'success': False, 'message': '姓名不能为空'})

    def _num(key, cast=float):
        v = (request.form.get(key) or '').strip()
        if not v:
            return None
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    age = _num('age', int)
    height_cm = _num('height_cm')
    weight_kg = _num('weight_kg')
    heart_rate = _num('heart_rate', int)
    spo2 = _num('spo2')

    db.execute_insert(
        """UPDATE patients SET
             name=%s, age=%s, gender=%s, contact_number=%s,
             emergency_contact=%s, emergency_phone=%s, emergency_info=%s,
             blood_type=%s, occupation=%s, address=%s,
             medical_history=%s, family_history=%s, allergies=%s,
             current_medications=%s,
             diagnosis_date=%s, disease_type=%s, diagnosis_hospital=%s,
             smoking_history=%s, occupational_exposure=%s,
             height_cm=%s, weight_kg=%s, blood_pressure=%s, heart_rate=%s, spo2=%s,
             updated_at=NOW()
           WHERE patient_id=%s""",
        (name, age, request.form.get('gender'), request.form.get('contact_number'),
         request.form.get('emergency_contact'), request.form.get('emergency_phone'),
         request.form.get('emergency_info'), request.form.get('blood_type'),
         request.form.get('occupation'), request.form.get('address'),
         request.form.get('medical_history'), request.form.get('family_history'),
         request.form.get('allergies'), request.form.get('current_medications'),
         request.form.get('diagnosis_date') or None, request.form.get('disease_type'),
         request.form.get('diagnosis_hospital'), request.form.get('smoking_history'),
         request.form.get('occupational_exposure'),
         height_cm, weight_kg, request.form.get('blood_pressure'), heart_rate, spo2,
         patient_id))

    # 与 users 表 full_name 保持同步
    db.execute_insert("UPDATE users SET full_name=%s WHERE user_id=%s", (name, current_user.id))

    db.disconnect()
    return jsonify({'success': True, 'message': '资料更新成功'})

# 患者趋势数据 API
@app.route('/api/patient/trend')
@login_required
def api_patient_trend():
    if current_user.user_type != 'patient':
        return jsonify({'error': '权限不足'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'error': '未找到患者信息'}), 404
    patient_id = patient_data[0]['patient_id']
    trend_data = db.get_patient_trend_data(patient_id) or []
    # JSON 序列化兼容处理（datetime -> str, Decimal -> float）
    for row in trend_data:
        if isinstance(row.get('created_at'), (datetime.datetime, datetime.date)):
            row['created_at'] = row['created_at'].strftime('%Y-%m-%d %H:%M')
        if isinstance(row.get('lesion_area_ratio'), Decimal):
            row['lesion_area_ratio'] = float(row['lesion_area_ratio'])
    db.disconnect()
    return jsonify(trend_data)

# 随访计划相关 API
@app.route('/api/followup', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
def followup_api():
    if current_user.user_type != 'patient':
        return jsonify({'error': '仅患者可操作随访计划'}), 403
    
    # 获取当前患者的 patient_id
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'error': '未找到患者信息'}), 404
    patient_id = patient_data[0]['patient_id']
    
    if request.method == 'GET':
        # 获取患者的随访计划
        status = request.args.get('status')
        plans = db.get_followup_plans(patient_id, status)
        db.disconnect()
        return jsonify(plans)
    
    elif request.method == 'POST':
        # 创建新的随访计划
        data = request.json
        suggested_date = data.get('suggested_date')
        notes = data.get('notes')
        if not suggested_date:
            return jsonify({'error': '建议日期不能为空'}), 400
        plan_id = db.create_followup_plan(patient_id, suggested_date, notes)
        db.disconnect()
        if plan_id:
            return jsonify({'success': True, 'plan_id': plan_id})
        else:
            return jsonify({'error': '创建失败'}), 500
    
    elif request.method == 'PUT':
        # 更新随访计划状态
        data = request.json
        plan_id = data.get('plan_id')
        status = data.get('status')
        if not plan_id or status not in ['pending', 'completed', 'cancelled']:
            return jsonify({'error': '参数无效'}), 400
        # 验证计划属于当前患者
        plan = db.execute_query("SELECT * FROM followup_plans WHERE plan_id = %s AND patient_id = %s", (plan_id, patient_id))
        if not plan:
            db.disconnect()
            return jsonify({'error': '计划不存在或无权操作'}), 404
        result = db.update_followup_status(plan_id, status)
        db.disconnect()
        return jsonify({'success': result is not None})
    
    elif request.method == 'DELETE':
        plan_id = request.args.get('plan_id')
        if not plan_id:
            return jsonify({'error': '缺少 plan_id'}), 400
        # 验证权限
        plan = db.execute_query("SELECT * FROM followup_plans WHERE plan_id = %s AND patient_id = %s", (plan_id, patient_id))
        if not plan:
            db.disconnect()
            return jsonify({'error': '计划不存在或无权操作'}), 404
        result = db.delete_followup_plan(plan_id)
        db.disconnect()
        return jsonify({'success': result is not None})


# ==================== 医生端随访管理 ====================
@app.route('/doctor/followup')
@login_required
def doctor_followup():
    """医生端随访管理页面：查看/创建/维护全部患者的随访计划"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('doctor_followup.html', plans=[], patients=[])
    plans = db.get_all_followup_plans() or []
    for p in plans:
        if isinstance(p.get('suggested_date'), datetime.date):
            p['suggested_date'] = p['suggested_date'].strftime('%Y-%m-%d')
        if isinstance(p.get('created_at'), datetime.datetime):
            p['created_at'] = p['created_at'].strftime('%Y-%m-%d %H:%M')
    patients = db.get_patients() or []
    db.disconnect()
    return render_template('doctor_followup.html', plans=plans, patients=patients)


@app.route('/api/doctor/followup', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
def doctor_followup_api():
    """医生端随访计划管理 API（GET 查询 / POST 创建 / PUT 状态 / DELETE 删除）"""
    if current_user.user_type != 'doctor':
        return jsonify({'error': '仅医生可操作'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500

    if request.method == 'GET':
        status = request.args.get('status')
        plans = db.get_all_followup_plans(status) or []
        for p in plans:
            if isinstance(p.get('suggested_date'), datetime.date):
                p['suggested_date'] = p['suggested_date'].strftime('%Y-%m-%d')
            if isinstance(p.get('created_at'), datetime.datetime):
                p['created_at'] = p['created_at'].strftime('%Y-%m-%d %H:%M')
        db.disconnect()
        return jsonify(plans)

    elif request.method == 'POST':
        data = request.json or {}
        patient_id = data.get('patient_id')
        suggested_date = data.get('suggested_date')
        notes = data.get('notes')
        if not patient_id or not suggested_date:
            return jsonify({'error': '患者和建议日期不能为空'}), 400
        # 校验患者存在且未删除
        patient = db.get_patient(patient_id)
        if not patient:
            db.disconnect()
            return jsonify({'error': '患者不存在'}), 404
        try:
            datetime.datetime.strptime(suggested_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': '日期格式无效'}), 400
        plan_id = db.create_followup_plan(patient_id, suggested_date, notes)
        # 通知患者
        if plan_id and patient[0].get('user_id'):
            db.add_notification(
                patient[0]['user_id'],
                '新的随访计划',
                f'医生已为您安排随访复查，建议日期：{suggested_date}。',
                'followup',
                '/patient/followup'
            )
        db.disconnect()
        if plan_id:
            return jsonify({'success': True, 'plan_id': plan_id})
        return jsonify({'error': '创建失败'}), 500

    elif request.method == 'PUT':
        data = request.json or {}
        plan_id = data.get('plan_id')
        status = data.get('status')
        if not plan_id or status not in ['pending', 'completed', 'cancelled']:
            return jsonify({'error': '参数无效'}), 400
        plan = db.get_followup_plan(plan_id)
        if not plan:
            db.disconnect()
            return jsonify({'error': '计划不存在'}), 404
        result = db.update_followup_status(plan_id, status)
        db.disconnect()
        return jsonify({'success': result is not None})

    elif request.method == 'DELETE':
        plan_id = request.args.get('plan_id')
        if not plan_id:
            return jsonify({'error': '缺少 plan_id'}), 400
        plan = db.get_followup_plan(plan_id)
        if not plan:
            db.disconnect()
            return jsonify({'error': '计划不存在'}), 404
        result = db.delete_followup_plan(plan_id)
        db.disconnect()
        return jsonify({'success': result is not None})


# ==================== 患者健康日志（health_logs）====================
@app.route('/patient/health_log')
@login_required
def patient_health_log():
    """患者健康日志页面：记录症状评分、用药、日常情况"""
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_health_log.html', logs=[], patient_id=None)
    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到患者信息', 'warning')
        return render_template('patient_health_log.html', logs=[], patient_id=None)
    patient_id = patient_data[0]['patient_id']
    logs = db.get_health_logs(patient_id) or []
    for log in logs:
        if isinstance(log.get('log_date'), datetime.date):
            log['log_date'] = log['log_date'].strftime('%Y-%m-%d')
        if isinstance(log.get('created_at'), datetime.datetime):
            log['created_at'] = log['created_at'].strftime('%Y-%m-%d %H:%M')
    db.disconnect()
    return render_template('patient_health_log.html', logs=logs, patient_id=patient_id)


@app.route('/api/health_logs', methods=['GET', 'POST', 'PUT', 'DELETE'])
@login_required
def health_logs_api():
    """患者健康日志 API（GET 查询 / POST 新增 / PUT 更新 / DELETE 删除）"""
    if current_user.user_type != 'patient':
        return jsonify({'error': '仅患者可操作健康日志'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'error': '未找到患者信息'}), 404
    patient_id = patient_data[0]['patient_id']

    if request.method == 'GET':
        logs = db.get_health_logs(patient_id) or []
        for log in logs:
            if isinstance(log.get('log_date'), datetime.date):
                log['log_date'] = log['log_date'].strftime('%Y-%m-%d')
            if isinstance(log.get('created_at'), datetime.datetime):
                log['created_at'] = log['created_at'].strftime('%Y-%m-%d %H:%M')
        db.disconnect()
        return jsonify(logs)

    elif request.method == 'POST':
        data = request.json or {}
        log_date = data.get('log_date')
        symptom_score = data.get('symptom_score')
        medication = data.get('medication')
        notes = data.get('notes')
        if not log_date:
            return jsonify({'error': '记录日期不能为空'}), 400
        try:
            datetime.datetime.strptime(log_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': '日期格式无效'}), 400
        # 症状评分 1-10 校验
        if symptom_score not in (None, ''):
            try:
                symptom_score = int(symptom_score)
                if not (1 <= symptom_score <= 10):
                    return jsonify({'error': '症状评分需在 1-10 之间'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': '症状评分格式无效'}), 400
        else:
            symptom_score = None
        log_id = db.add_health_log(patient_id, log_date, symptom_score, medication, notes)
        db.disconnect()
        if log_id:
            return jsonify({'success': True, 'log_id': log_id})
        return jsonify({'error': '创建失败'}), 500

    elif request.method == 'PUT':
        data = request.json or {}
        log_id = data.get('log_id')
        log_date = data.get('log_date')
        symptom_score = data.get('symptom_score')
        medication = data.get('medication')
        notes = data.get('notes')
        if not log_id or not log_date:
            return jsonify({'error': '参数缺失'}), 400
        log = db.get_health_log(patient_id, log_id)
        if not log:
            db.disconnect()
            return jsonify({'error': '记录不存在或无权操作'}), 404
        try:
            datetime.datetime.strptime(log_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': '日期格式无效'}), 400
        if symptom_score not in (None, ''):
            try:
                symptom_score = int(symptom_score)
                if not (1 <= symptom_score <= 10):
                    return jsonify({'error': '症状评分需在 1-10 之间'}), 400
            except (ValueError, TypeError):
                return jsonify({'error': '症状评分格式无效'}), 400
        else:
            symptom_score = None
        result = db.update_health_log(log_id, log_date, symptom_score, medication, notes)
        db.disconnect()
        return jsonify({'success': result is not None})

    elif request.method == 'DELETE':
        log_id = request.args.get('log_id')
        if not log_id:
            return jsonify({'error': '缺少 log_id'}), 400
        log = db.get_health_log(patient_id, log_id)
        if not log:
            db.disconnect()
            return jsonify({'error': '记录不存在或无权操作'}), 404
        result = db.delete_health_log(log_id)
        db.disconnect()
        return jsonify({'success': result is not None})


# API: 患者健康日志症状评分趋势（供趋势分析页使用）
@app.route('/api/patient/health_trend')
@login_required
def api_patient_health_trend():
    if current_user.user_type != 'patient':
        return jsonify({'error': '权限不足'}), 403
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'error': '未找到患者信息'}), 404
    trend = db.get_health_log_trend(patient_data[0]['patient_id']) or []
    for row in trend:
        if isinstance(row.get('log_date'), datetime.date):
            row['log_date'] = row['log_date'].strftime('%Y-%m-%d')
        if row.get('symptom_score') is not None:
            row['symptom_score'] = int(row['symptom_score'])
    db.disconnect()
    return jsonify(trend)


# ==================== 消息通知 ====================
@app.route('/api/notifications')
@login_required
def api_notifications():
    """获取当前用户的通知列表"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    notifications = db.get_notifications(current_user.id, limit=20)
    unread = db.get_unread_notification_count(current_user.id)
    db.disconnect()
    return jsonify({
        'notifications': notifications or [],
        'unread_count': unread
    })


@app.route('/api/notifications/unread')
@login_required
def api_notifications_unread():
    """获取当前用户的未读通知数量"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    unread = db.get_unread_notification_count(current_user.id)
    db.disconnect()
    return jsonify({'unread_count': unread})


@app.route('/api/notifications/<int:notification_id>/read', methods=['POST'])
@login_required
def api_notification_read(notification_id):
    """标记单条通知为已读"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    result = db.mark_notification_read(notification_id, current_user.id)
    db.disconnect()
    if result is not None:
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': '通知不存在或无权操作'}), 404


@app.route('/api/notifications/read_all', methods=['POST'])
@login_required
def api_notifications_read_all():
    """标记当前用户全部通知为已读"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    result = db.mark_all_notifications_read(current_user.id)
    db.disconnect()
    return jsonify({'success': result is not None})


# ==================== 在线咨询 / 聊天 ====================
@app.route('/doctor/chat')
@login_required
def doctor_chat():
    """医生端在线咨询页面"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    return render_template('doctor_chat.html')


@app.route('/api/chat/conversations', methods=['GET'])
@login_required
def api_chat_conversations():
    """获取当前用户的会话列表（含未读数和对方信息）"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    conversations = db.get_user_conversations(current_user.id, current_user.user_type) or []
    # 附加每个会话的未读数
    for conv in conversations:
        conv['unread_count'] = db.get_conversation_unread_count(
            conv['conversation_id'], current_user.id)
    db.disconnect()
    return jsonify({'conversations': conversations})


@app.route('/api/chat/conversations', methods=['POST'])
@login_required
def api_chat_start():
    """患者发起与某位医生的会话（若已有进行中会话则复用）"""
    if current_user.user_type != 'patient':
        return jsonify({'error': '仅患者可发起咨询'}), 403
    data = request.json or {}
    doctor_id = data.get('doctor_id')
    if not doctor_id:
        return jsonify({'error': '缺少 doctor_id'}), 400
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    conv = db.find_conversation(current_user.id, doctor_id)
    if not conv:
        conv_id = db.create_conversation(current_user.id, doctor_id)
        if not conv_id:
            db.disconnect()
            return jsonify({'error': '创建会话失败'}), 500
    else:
        conv_id = conv[0]['conversation_id']
    db.disconnect()
    return jsonify({'success': True, 'conversation_id': conv_id})


@app.route('/api/chat/conversations/<int:conversation_id>/messages', methods=['GET'])
@login_required
def api_chat_messages(conversation_id):
    """获取会话消息，同时把发给当前用户的消息标记为已读"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    conv = db.get_conversation(conversation_id)
    if not conv:
        db.disconnect()
        return jsonify({'error': '会话不存在'}), 404
    conv = conv[0]
    # 权限检查：仅会话双方可查看
    if current_user.id not in (conv['patient_id'], conv['doctor_id']):
        db.disconnect()
        return jsonify({'error': '无权访问'}), 403
    messages = db.get_conversation_messages(conversation_id) or []
    # 标记当前用户收到的消息为已读
    db.mark_conversation_read(conversation_id, current_user.id)
    # 附上对方信息
    peer_id = conv['doctor_id'] if current_user.id == conv['patient_id'] else conv['patient_id']
    peer = db.execute_query(
        "SELECT user_id, full_name, role, specialization FROM users WHERE user_id = %s",
        (peer_id,)
    )
    db.disconnect()
    return jsonify({
        'conversation': conv,
        'peer': peer[0] if peer else None,
        'messages': messages
    })


@app.route('/api/chat/conversations/<int:conversation_id>/messages', methods=['POST'])
@login_required
def api_chat_send(conversation_id):
    """发送一条聊天消息"""
    data = request.json or {}
    content = (data.get('content') or '').strip()
    if not content:
        return jsonify({'error': '消息内容不能为空'}), 400
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    conv = db.get_conversation(conversation_id)
    if not conv:
        db.disconnect()
        return jsonify({'error': '会话不存在'}), 404
    conv = conv[0]
    if current_user.id not in (conv['patient_id'], conv['doctor_id']):
        db.disconnect()
        return jsonify({'error': '无权操作'}), 403
    receiver_id = conv['doctor_id'] if current_user.id == conv['patient_id'] else conv['patient_id']
    message_id = db.add_chat_message(conversation_id, current_user.id, receiver_id, content)
    db.disconnect()
    if message_id:
        return jsonify({'success': True, 'message_id': message_id})
    return jsonify({'error': '发送失败'}), 500


@app.route('/api/chat/unread')
@login_required
def api_chat_unread():
    """获取当前用户的聊天未读消息总数（用于角标）"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    count = db.get_total_unread_chat_count(current_user.id)
    db.disconnect()
    return jsonify({'unread_count': count})


@app.route('/api/chat/conversations/<int:conversation_id>/close', methods=['POST'])
@login_required
def api_chat_close(conversation_id):
    """结束会话"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    conv = db.get_conversation(conversation_id)
    if not conv:
        db.disconnect()
        return jsonify({'error': '会话不存在'}), 404
    conv = conv[0]
    if current_user.id not in (conv['patient_id'], conv['doctor_id']):
        db.disconnect()
        return jsonify({'error': '无权操作'}), 403
    result = db.close_conversation(conversation_id)
    db.disconnect()
    return jsonify({'success': result is not None})


if __name__ == '__main__':
    # 确保必要目录存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./static/images'):
        os.makedirs('./static/images')
    gradcam_dir = os.path.join('static', 'gradcam')
    if not os.path.exists(gradcam_dir):
        os.makedirs(gradcam_dir)
    
    # Windows + Anaconda 环境下 debug reloader 会误监视 site-packages 导致无限重启崩溃，
    # 故关闭自动重载（use_reloader=False），仍保留 debug 详细错误页；代码修改后手动重启即可
    app.run(debug=True, use_reloader=False)
