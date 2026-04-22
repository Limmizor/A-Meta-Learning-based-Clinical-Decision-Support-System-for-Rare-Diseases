from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import uuid
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from database import Database
from pf_diagnosis_service import PFDianosisService
import datetime
import pydicom
from PIL import Image
import numpy as np


app = Flask(__name__)
app.config.from_object(Config)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 应使用强密钥

# 初始化Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'maml_model_final .pth')
pf_service = PFDianosisService(model_path=model_path)

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

        # 跳转
        if user['role'] == 'doctor':
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
        return render_template('my_reports.html', reports=[])
    
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到关联的患者信息', 'danger')
        return render_template('my_reports.html', reports=[])
    
    patient_id = patient_data[0]['patient_id']
    reports = db.get_diagnosis_reports(patient_id)
    db.disconnect()
    return render_template('my_reports.html', reports=reports)

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
    
    model_trained = os.path.exists('./models/pf_maml_model.pth')
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('doctor_dashboard.html', patients=[], model_trained=model_trained)
    
    patients = db.get_patients()
    db.disconnect()
    return render_template('doctor_dashboard.html', patients=patients, model_trained=model_trained)

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
    
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到关联的患者信息', 'danger')
        return render_template('patient_dashboard.html', reports=[], patient=None)
    
    patient_id = patient_data[0]['patient_id']
    reports = db.get_diagnosis_reports(patient_id)
    patient = db.get_patient(patient_id)
    db.disconnect()
    
    return render_template('patient_dashboard.html', 
                         reports=reports, 
                         patient=patient[0] if patient else None)

@app.route('/patient/trend')
@login_required
def patient_trend():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    return render_template('patient_trend.html')

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
        if current_user.user_type == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))
    return redirect(url_for('login'))

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'dcm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    
    db.disconnect()
    model_trained = os.path.exists('./models/pf_maml_model.pth')
    return render_template('patient.html', 
                          patient=patient[0], 
                          images=images, 
                          reports=reports,
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
    
    # 调用诊断服务（返回 predictions 和量化指标）
    # 注意：原 pf_service.diagnose_patient 只返回 predictions，我们需要扩展它
    # 临时方案：先获取 predictions，然后生成模拟量化指标
    predictions = pf_service.diagnose_patient(patient_id)
    
    # 模拟量化指标（实际应从模型获取）
    lesion_area_ratio = 0.32
    distribution_range = '双肺下叶背段及胸膜下'
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    doctor_id = current_user.id
    report_id = db.add_diagnosis_report(
        patient_id, doctor_id, clinical_notes, "AI辅助诊断结果",
        lesion_area_ratio=lesion_area_ratio,
        distribution_range=distribution_range
    )
    
    if report_id:
        for pred in predictions:
            db.add_disease_prediction(report_id, pred['disease_id'], pred['confidence'], pred['rank'])
        db.disconnect()
        return jsonify({'success': True, 'message': '诊断完成', 'report_id': report_id})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '创建诊断报告失败'})

# 模型训练接口（已禁用，返回提示）
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
    if current_user.user_type == 'doctor':
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
@app.route('/update_patient', methods=['POST'])
@login_required
def update_patient():
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    patient_id = request.form.get('patient_id')
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    contact_number = request.form.get('contact_number')
    medical_history = request.form.get('medical_history')
    if not patient_id or not name:
        return jsonify({'success': False, 'message': '患者ID和姓名不能为空'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    # 更新患者信息（注意：原 add_patient 方法没有提供更新函数，这里直接执行 SQL）
    try:
        update_query = """
            UPDATE patients 
            SET name=%s, age=%s, gender=%s, contact_number=%s, medical_history=%s, updated_at=%s
            WHERE patient_id=%s
        """
        params = (name, age, gender, contact_number, medical_history, datetime.datetime.now(), patient_id)
        db.execute_insert(update_query, params)
        db.disconnect()
        return jsonify({'success': True, 'message': '患者信息更新成功'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'更新失败: {str(e)}'})



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
    if patient_data:
        patient_id = patient_data[0]['patient_id']
        reports = db.get_diagnosis_reports(patient_id)
        reports_count = len(reports) if reports else 0
    db.disconnect()
    
    return render_template('patient_profile.html', 
                         patient=patient_data[0] if patient_data else None,
                         user=current_user,
                         reports_count=reports_count,
                         appointments_count=0,
                         prescriptions_count=0,
                         follow_up_count=0)

# API: 获取所有疾病（用于疾病查询页面）
@app.route('/api/diseases')
@login_required
def api_diseases():
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    diseases = db.get_diseases()
    db.disconnect()
    return jsonify(diseases)

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
    return jsonify(disease[0])

# API: 创建新疾病（仅医生）
@app.route('/api/diseases', methods=['POST'])
@login_required
def api_create_disease():
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '权限不足'}), 403
    data = request.json
    name = data.get('name')
    if not name:
        return jsonify({'success': False, 'message': '疾病名称不能为空'}), 400
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'}), 500
    disease_id = db.execute_insert(
        """INSERT INTO diseases (name, icd_code, description, symptoms, diagnostic_criteria, 
           treatment_options, prevention, is_featured) 
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (name, data.get('icd_code'), data.get('description'), data.get('symptoms'),
         data.get('diagnostic_criteria'), data.get('treatment_options'),
         data.get('prevention'), data.get('is_featured', False))
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
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '权限不足'}), 403
    data = request.json
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'}), 500
    result = db.execute_insert(
        """UPDATE diseases SET name=%s, icd_code=%s, description=%s, symptoms=%s, 
           diagnostic_criteria=%s, treatment_options=%s, prevention=%s, is_featured=%s 
           WHERE disease_id=%s""",
        (data.get('name'), data.get('icd_code'), data.get('description'), data.get('symptoms'),
         data.get('diagnostic_criteria'), data.get('treatment_options'),
         data.get('prevention'), data.get('is_featured', False), disease_id)
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
    if current_user.user_type != 'doctor':
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
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
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
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
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
    if current_user.user_type != 'doctor':
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
@app.route('/delete_patient/<int:patient_id>', methods=['DELETE'])
@login_required
def delete_patient(patient_id):
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    try:
        patient_data = db.execute_query("SELECT name FROM patients WHERE patient_id = %s", (patient_id,))
        patient_name = patient_data[0]['name'] if patient_data else '未知患者'
        result = db.execute_insert("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
        if result:
            db.add_system_log(current_user.id, 'DELETE_PATIENT', f'删除患者: {patient_name}')
            db.disconnect()
            return jsonify({'success': True, 'message': '患者删除成功'})
        else:
            db.disconnect()
            return jsonify({'success': False, 'message': '删除患者失败'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'删除异常: {str(e)}'})

# 医生今日日程
@app.route('/doctor/schedule')
@login_required
def doctor_schedule():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    return render_template('doctor_schedule.html')

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
    diseases = db.get_diseases()
    featured_diseases = db.execute_query(
        "SELECT * FROM diseases WHERE is_featured = 1 ORDER BY created_at DESC LIMIT 6"
    ) or []
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
        "SELECT user_id, full_name, specialty, title, department FROM users WHERE role = 'doctor'"
    ) or []
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    appointments = []
    if patient_data:
        patient_id = patient_data[0]['patient_id']
        appointments = db.execute_query(
            """SELECT a.*, u.full_name as doctor_name, u.specialty, u.department 
               FROM appointments a 
               JOIN users u ON a.doctor_id = u.user_id 
               WHERE a.patient_id = %s 
               ORDER BY a.appointment_date DESC, a.appointment_time DESC""",
            (patient_id,)
        ) or []
    db.disconnect()
    return render_template('patient_appointment.html', doctors=doctors, appointments=appointments)

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
    result = db.execute_insert(
        "UPDATE appointments SET status = 'cancelled', updated_at = %s WHERE appointment_id = %s",
        (datetime.datetime.now(), appointment_id)
    )
    if result:
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
        return render_template('patient_chat.html', patient=None, messages=[])
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    patient = patient_data[0] if patient_data else None
    messages = [
        {
            'id': 1,
            'sender': 'doctor',
            'sender_name': '张医生',
            'content': '您好，请问有什么可以帮助您的？',
            'timestamp': '2024-01-15 10:30:00',
            'avatar': 'D'
        },
        {
            'id': 2,
            'sender': 'patient',
            'sender_name': '我',
            'content': '我最近感觉关节有些疼痛，特别是早上起床时。',
            'timestamp': '2024-01-15 10:32:15',
            'avatar': 'P'
        },
        {
            'id': 3,
            'sender': 'doctor',
            'sender_name': '张医生',
            'content': '这种情况持续多久了？有没有其他症状？',
            'timestamp': '2024-01-15 10:33:45',
            'avatar': 'D'
        }
    ]
    online_doctors = db.execute_query(
        "SELECT user_id, full_name, specialty, department FROM users WHERE role = 'doctor' LIMIT 5"
    ) or []
    db.disconnect()
    return render_template('patient_chat.html', 
                         patient=patient,
                         messages=messages,
                         online_doctors=online_doctors)

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
        thumbnails = []  # 存储缩略图 URL
        
        # 确保缩略图目录存在
        thumb_dir = os.path.join('static', 'thumbnails')
        os.makedirs(thumb_dir, exist_ok=True)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # 生成缩略图
                thumb_filename = f"thumb_{unique_filename}.png"
                thumb_path = os.path.join(thumb_dir, thumb_filename)
                
                ext = os.path.splitext(unique_filename)[1].lower()
                if ext == '.dcm':
                    # DICOM 转 PNG
                    dcm = pydicom.dcmread(filepath)
                    img = dcm.pixel_array.astype(np.float32)
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    img = (img * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img).convert('RGB')
                else:
                    img_pil = Image.open(filepath).convert('RGB')
                # 生成缩略图（150x150 以内）
                img_pil.thumbnail((150, 150))
                img_pil.save(thumb_path)
                
                thumbnail_url = f'/static/thumbnails/{thumb_filename}'
                thumbnails.append(thumbnail_url)
                
                # 保存到数据库（存储原始文件名）
                db.add_medical_image(patient_id, unique_filename, 'CT', 'AI诊断上传')
                saved_paths.append(filepath)
        
        db.disconnect()
        if not saved_paths:
            return jsonify({'success': False, 'message': '文件上传失败'})
        
        # 调用诊断服务
        predictions, heatmap_url, lesion_ratio, distribution, findings, suggestions = \
            pf_service.predict_from_paths(saved_paths, patient_id)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'heatmap_url': heatmap_url,
            'lesion_area_ratio': lesion_ratio,
            'distribution_range': distribution,
            'imaging_findings': findings,
            'suggestions': suggestions,
            'time_cost': 32,
            'thumbnails': thumbnails   # 返回缩略图 URL 列表
        })
    except Exception as e:
        print(f"AI诊断错误: {e}")
        return jsonify({'success': False, 'message': 'AI诊断服务暂时不可用，请稍后重试'})

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
    
    # 如果是医生，可能还有 specialty 等字段，如果表中有则更新
    if current_user.user_type == 'doctor' and specialty:
        # 假设 users 表有 specialty 字段，如果没有请先 ALTER TABLE users ADD COLUMN specialty VARCHAR(100)
        db.execute_insert("UPDATE users SET specialty=%s WHERE user_id=%s", (specialty, user_id))
    
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
    
    # 获取患者记录
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id=%s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到患者信息'})
    
    patient_id = patient_data[0]['patient_id']
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    contact_number = request.form.get('contact_number')
    medical_history = request.form.get('medical_history')
    # 可选：血型、紧急联系人等（如果表中有相应字段）
    
    update_query = """
        UPDATE patients SET name=%s, age=%s, gender=%s, contact_number=%s, medical_history=%s
        WHERE patient_id=%s
    """
    db.execute_insert(update_query, (name, age, gender, contact_number, medical_history, patient_id))
    
    # 同时更新 users 表的 full_name（与患者姓名同步）
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
    patient_data = db.execute_query("SELECT patient_id FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'error': '未找到患者信息'}), 404
    patient_id = patient_data[0]['patient_id']
    trend_data = db.get_patient_trend_data(patient_id)
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
    
    app.run(debug=True)