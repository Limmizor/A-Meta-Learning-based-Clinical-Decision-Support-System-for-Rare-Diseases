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
pf_service = PFDianosisService(model_path='./models/pf_maml_model.pth')

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
        db.disconnect()
        
        if not user_data or not check_password_hash(user_data[0]['password_hash'], password):
            flash('用户名或密码错误', 'danger')
            return render_template('login.html')
        
        user = User(user_data[0]['user_id'], user_data[0]['username'], 
                   user_data[0]['role'], user_data[0]['full_name'])
        login_user(user, remember=remember)
        
        if user.user_type == 'doctor':
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
        
        hashed_password = generate_password_hash(password)
        user_id = db.execute_insert(
            "INSERT INTO users (username, password_hash, email, full_name, role) VALUES (%s, %s, %s, %s, %s)",
            (username, hashed_password, email, full_name, user_type)
        )
        
        if user_type == 'patient':
            patient_name = request.form.get('patient_name') or full_name
            patient_gender = request.form.get('patient_gender')
            patient_age = request.form.get('patient_age')
            db.execute_insert(
                "INSERT INTO patients (name, age, gender, user_id) VALUES (%s, %s, %s, %s)",
                (patient_name, patient_age, patient_gender, user_id)
            )
        
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

# 诊断接口（原患者详情页使用，使用肺纤维化模型）
@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    patient_id = request.form.get('patient_id')
    clinical_notes = request.form.get('clinical_notes')
    
    if not patient_id:
        return jsonify({'success': False, 'message': '患者ID不能为空'})
    
    # 调用肺纤维化诊断服务
    predictions = pf_service.diagnose_patient(patient_id)
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    # 假设当前医生ID为2（可根据实际情况获取）
    doctor_id = current_user.id
    report_id = db.add_diagnosis_report(patient_id, doctor_id, clinical_notes, "AI辅助诊断结果")
    
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

# 疾病管理页面（旧版，保留但可能不再使用）
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
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            db.add_medical_image(patient_id, unique_filename, 'CT', 'AI诊断上传')
            saved_paths.append(filepath)
    db.disconnect()
    if not saved_paths:
        return jsonify({'success': False, 'message': '文件上传失败'})
    
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
        'time_cost': 32
    })

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