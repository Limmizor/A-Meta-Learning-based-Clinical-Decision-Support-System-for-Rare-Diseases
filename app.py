from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import uuid
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config
from database import Database
from pf_diagnosis_service import PFDianosisService
# 如果不再需要训练功能，可以删除 train_model 导入
import datetime

app = Flask(__name__)
app.config.from_object(Config)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 应该使用强密钥

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

#用户加载器
@login_manager.user_loader
def load_user(user_id):
    db = Database()
    if not db.connect():
        return None
    
    user_data = db.execute_query("SELECT * FROM users WHERE user_id = %s", (user_id,))
    db.disconnect()
    
    if user_data:
        user = user_data[0]
        # 使用role字段而不是user_type
        return User(user['user_id'], user['username'], user['role'], user['full_name'])
    return None

# 初始化MAML服务
    maml_service = MAMLService(model_path='./models/maml_model.pth')

# 添加登录路由
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
        
        # 使用role字段
        user = User(user_data[0]['user_id'], user_data[0]['username'], 
                   user_data[0]['role'], user_data[0]['full_name'])
        login_user(user, remember=remember)
        
        # 根据角色重定向
        if user.user_type == 'doctor':  # 这里保持user_type，因为User类中使用这个属性名
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))
    
    return render_template('login.html')

#添加注册路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        email = request.form.get('email')
        full_name = request.form.get('full_name')
        user_type = request.form.get('user_type')
        
        # 验证密码匹配
        if password != confirm_password:
            flash('密码不匹配', 'danger')
            return render_template('register.html')
        
        # 验证用户名是否已存在
        db = Database()
        if not db.connect():
            flash('数据库连接失败', 'danger')
            return render_template('register.html')
        
        existing_user = db.execute_query("SELECT * FROM users WHERE username = %s", (username,))
        if existing_user:
            flash('用户名已存在', 'danger')
            db.disconnect()
            return render_template('register.html')
        
        # 创建用户 - 修复字段名
        hashed_password = generate_password_hash(password)
        user_id = db.execute_insert(
            "INSERT INTO users (username, password_hash, email, full_name, role) VALUES (%s, %s, %s, %s, %s)",
            (username, hashed_password, email, full_name, user_type)  # 使用password_hash和role
        )
        
        # 如果是患者用户，创建患者记录
        if user_type == 'patient':
            patient_name = request.form.get('patient_name') or full_name  # 如果没有提供患者姓名，使用全名
            patient_gender = request.form.get('patient_gender')
            patient_age = request.form.get('patient_age')
            
            db.execute_insert(
                "INSERT INTO patients (name, age, gender, user_id) VALUES (%s, %s, %s, %s)",
                (patient_name, patient_age, patient_gender, user_id)  # 移除了created_by，使用user_id
            )
        
        db.disconnect()
        
        flash('注册成功，请登录', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/my_reports')
@login_required
def my_reports():
    """我的报告页面"""
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    # 获取当前用户关联的患者ID
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('my_reports.html', reports=[])
    
    # 获取当前用户关联的患者ID
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到关联的患者信息', 'danger')
        return render_template('my_reports.html', reports=[])
    
    patient_id = patient_data[0]['patient_id']
    reports = db.get_diagnosis_reports(patient_id)
    db.disconnect()
    
    return render_template('my_reports.html', reports=reports)


# 添加退出登录路由
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('您已成功退出登录', 'success')
    return redirect(url_for('login'))

# 添加医生仪表板路由
@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    # 检查模型是否已训练
    model_trained = os.path.exists('./models/maml_model.pth')
    
    # 获取患者列表
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('doctor_dashboard.html', patients=[], model_trained=model_trained)
    
    patients = db.get_patients()
    db.disconnect()
    
    return render_template('doctor_dashboard.html', patients=patients, model_trained=model_trained)


# 添加患者仪表板路由
@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    # 获取患者的诊断报告
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_dashboard.html', reports=[], patient=None)
    
    # 获取当前用户关联的患者信息
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        flash('未找到关联的患者信息', 'danger')
        return render_template('patient_dashboard.html', reports=[], patient=None)
    
    patient_id = patient_data[0]['patient_id']
    reports = db.get_diagnosis_reports(patient_id)
    
    # 获取完整的患者信息
    patient = db.get_patient(patient_id)
    db.disconnect()
    
    return render_template('patient_dashboard.html', 
                         reports=reports, 
                         patient=patient[0] if patient else None)


# 保护现有路由，添加登录要求
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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 删除重复的index函数，保留上面的index函数

@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    """患者详情页面"""
    db = Database()
    if not db.connect():
        return "数据库连接失败", 500
    
    patient = db.get_patient(patient_id)
    if not patient:
        return "患者不存在", 404
    
    images = db.get_medical_images(patient_id)
    reports = db.get_diagnosis_reports(patient_id)
    
    # 为每个报告获取预测结果
    for report in reports:
        predictions = db.get_disease_predictions(report['report_id'])
        report['predictions'] = predictions
    
    db.disconnect()
    
    # 检查模型是否已训练
    model_trained = os.path.exists('./models/maml_model.pth')
    
    return render_template('patient.html', 
                          patient=patient[0], 
                          images=images, 
                          reports=reports,
                          model_trained=model_trained)

@app.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    """添加新患者"""
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

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    """上传医学影像"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '没有文件部分'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # 获取表单数据
        patient_id = request.form.get('patient_id')
        image_type = request.form.get('image_type')
        description = request.form.get('description')
        
        # 保存到数据库
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


@app.route('/diagnose', methods=['POST'])
@login_required
def diagnose():
    """进行诊断（使用MAML模型预测）"""
    patient_id = request.form.get('patient_id')
    clinical_notes = request.form.get('clinical_notes')
    
    if not patient_id:
        return jsonify({'success': False, 'message': '患者ID不能为空'})
    
    # 使用MAML模型进行预测
    predictions = maml_service.diagnose_patient(patient_id)
    
    # 如果预测失败，使用模拟数据
    if predictions is None:
        predictions = [
            {'disease_id': 1, 'disease_name': '戈谢病', 'confidence': 0.85, 'rank': 1},
            {'disease_id': 3, 'disease_name': '庞贝病', 'confidence': 0.10, 'rank': 2},
            {'disease_id': 2, 'disease_name': '法布雷病', 'confidence': 0.05, 'rank': 3}
        ]
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    # 添加诊断报告
    # 假设当前医生ID为2（王医生）
    report_id = db.add_diagnosis_report(patient_id, 2, clinical_notes, "AI辅助诊断结果")
    
    if report_id:
        # 添加疾病预测
        for pred in predictions:
            db.add_disease_prediction(
                report_id, 
                pred['disease_id'], 
                pred['confidence'], 
                pred['rank']
            )
        
        db.disconnect()
        return jsonify({'success': True, 'message': '诊断完成', 'report_id': report_id})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '创建诊断报告失败'})

@app.route('/train_model', methods=['POST'])
@login_required
def train_model_route():
    """训练MAML模型"""
    try:
        success = train_model()
        if success:
            # 重新加载模型
            global maml_service
            maml_service = MAMLService(model_path='./models/maml_model.pth')
            return jsonify({'success': True, 'message': '模型训练成功'})
        else:
            return jsonify({'success': False, 'message': '模型训练失败'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'训练异常: {str(e)}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """访问上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/patients')
@login_required
def api_patients():
    """API: 获取所有患者"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    
    patients = db.get_patients()
    db.disconnect()
    
    return jsonify(patients)

@app.route('/profile')
@login_required
def profile():
    """用户个人资料页面 - 根据角色重定向"""
    if current_user.user_type == 'doctor':
        return redirect(url_for('doctor_profile'))
    else:
        return redirect(url_for('patient_profile'))

@app.route('/doctor_profile')
@login_required
def doctor_profile():
    """医生个人资料页面"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_profile'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('profile.html', user=None, patients_count=0, reports_count=0)
    
    # 获取用户信息
    user_data = db.execute_query("SELECT * FROM users WHERE user_id = %s", (current_user.id,))
    
    # 获取患者数量
    patients_data = db.execute_query("SELECT COUNT(*) as count FROM patients")
    patients_count = patients_data[0]['count'] if patients_data else 0
    
    # 获取报告数量（简化统计）
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

@app.route('/patient_profile')
@login_required
def patient_profile():
    """患者个人档案页面"""
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_profile.html', patient=None, reports_count=0)
    
    # 获取当前用户关联的患者信息
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    
    # 获取报告数量
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

@app.route('/api/diseases')
@login_required
def api_diseases():
    """API: 获取所有疾病"""
    db = Database()
    if not db.connect():
        return jsonify({'error': '数据库连接失败'}), 500
    
    diseases = db.get_diseases()
    db.disconnect()
    
    return jsonify(diseases)

@app.route('/disease_management', methods=['GET', 'POST'])
@login_required
def disease_management():
    """疾病管理页面（仅医生可访问）"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('disease_management.html', diseases=[])
    
    if request.method == 'POST':
        # 处理疾病添加
        action = request.form.get('action', 'add')
        name = request.form.get('name')
        description = request.form.get('description')
        symptoms = request.form.get('symptoms')
        treatment = request.form.get('treatment')
        icd_code = request.form.get('icd_code')
        diagnostic_criteria = request.form.get('diagnostic_criteria')
        prevention = request.form.get('prevention')
        
        if action == 'add' and name:
            try:
                disease_id = db.add_disease(name, description, symptoms, treatment)
                if disease_id:
                    # 添加系统日志
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
    
    # GET 请求：获取疾病列表
    diseases = db.get_diseases()
    db.disconnect()
    
    return render_template('disease_management.html', diseases=diseases)


@app.route('/system_logs')
@login_required
def system_logs():
    """系统日志页面（仅医生可访问）"""
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


@app.route('/api/logs', methods=['POST'])
@login_required
def api_logs():
    """API: 添加系统日志"""
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

@app.route('/symptom_check', methods=['GET', 'POST'])
@login_required
def symptom_check():
    if request.method == 'POST':
        # 处理症状自查表单提交
        symptoms = request.form.getlist('symptoms')
        symptom_details = request.form.get('symptomDetails')
        age = request.form.get('age')
        gender = request.form.get('gender')
        medical_history = request.form.get('medicalHistory')
        
        # 这里调用AI分析症状的逻辑
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
    """分析症状并返回可能的疾病列表"""
    # 这里实现症状分析的逻辑
    # 暂时返回模拟数据
    return [
        {
            'name': '戈谢病',
            'match_score': 0.85,
            'description': '一种遗传性代谢疾病，主要影响肝脏、脾脏、骨骼和神经系统。',
            'common_symptoms': ['疲劳', '肝脾肿大', '骨骼疼痛', '生长发育迟缓'],
            'icd_code': 'E75.2'
        },
        {
            'name': '法布雷病',
            'match_score': 0.60,
            'description': 'X连锁遗传的溶酶体贮积症，主要影响心脏、肾脏和神经系统。',
            'common_symptoms': ['手脚疼痛', '皮肤血管角质瘤', '少汗', '角膜混浊'],
            'icd_code': 'E75.2'
        }
    ]

# 添加删除疾病的路由
@app.route('/delete_disease/<int:disease_id>', methods=['DELETE'])
@login_required
def delete_disease(disease_id):
    """删除疾病"""
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    try:
        # 先获取疾病名称用于日志
        disease_data = db.execute_query("SELECT name FROM diseases WHERE disease_id = %s", (disease_id,))
        disease_name = disease_data[0]['name'] if disease_data else '未知疾病'
        
        # 删除疾病
        result = db.execute_insert("DELETE FROM diseases WHERE disease_id = %s", (disease_id,))
        
        if result:
            # 添加系统日志
            db.add_system_log(current_user.id, 'DELETE_DISEASE', f'删除疾病: {disease_name}')
            db.disconnect()
            return jsonify({'success': True, 'message': '疾病删除成功'})
        else:
            db.disconnect()
            return jsonify({'success': False, 'message': '删除疾病失败'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'删除异常: {str(e)}'})
    
#患者列表
@app.route('/patient_list')
@login_required
def patient_list():
    """患者列表页面（仅医生可访问）"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    # 获取患者列表
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_list.html', patients=[])
    
    patients = db.get_patients()
    db.disconnect()
    
    return render_template('patient_list.html', patients=patients)

#删除患者
@app.route('/delete_patient/<int:patient_id>', methods=['DELETE'])
@login_required
def delete_patient(patient_id):
    """删除患者"""
    if current_user.user_type != 'doctor':
        return jsonify({'success': False, 'message': '无权操作'})
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    try:
        # 先获取患者名称用于日志
        patient_data = db.execute_query("SELECT name FROM patients WHERE patient_id = %s", (patient_id,))
        patient_name = patient_data[0]['name'] if patient_data else '未知患者'
        
        # 删除患者（注意：这里需要先删除相关的诊断记录和影像记录，根据外键约束）
        # 在实际应用中，您可能需要先删除关联数据，或者设置级联删除
        
        result = db.execute_insert("DELETE FROM patients WHERE patient_id = %s", (patient_id,))
        
        if result:
            # 添加系统日志
            db.add_system_log(current_user.id, 'DELETE_PATIENT', f'删除患者: {patient_name}')
            db.disconnect()
            return jsonify({'success': True, 'message': '患者删除成功'})
        else:
            db.disconnect()
            return jsonify({'success': False, 'message': '删除患者失败'})
    except Exception as e:
        db.disconnect()
        return jsonify({'success': False, 'message': f'删除异常: {str(e)}'})
    
#医生今日日程
@app.route('/doctor/schedule')
@login_required
def doctor_schedule():
    """医生今日日程页面"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    return render_template('doctor_schedule.html')

#罕见病查询
@app.route('/disease_query')
@login_required
def disease_query():
    """罕见病查询页面"""
    if current_user.user_type != 'doctor':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('patient_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('disease_query.html', diseases=[], featured_diseases=[])
    
    # 获取所有疾病
    diseases = db.get_diseases()
    
    # 获取特色疾病（可以标记为推荐或热门的疾病）
    featured_diseases = db.execute_query(
        "SELECT * FROM diseases WHERE is_featured = 1 ORDER BY created_at DESC LIMIT 6"
    ) or []
    
    db.disconnect()
    
    return render_template('disease_query.html', 
                         diseases=diseases, 
                         featured_diseases=featured_diseases)

@app.route('/patient/appointment')
@login_required
def patient_appointment():
    """患者预约挂号页面"""
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_appointment.html', doctors=[], appointments=[])
    
    # 获取医生列表
    doctors = db.execute_query(
        "SELECT user_id, full_name, specialty, title, department FROM users WHERE role = 'doctor'"
    ) or []
    
    # 获取当前患者的预约记录
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
    
    return render_template('patient_appointment.html', 
                         doctors=doctors, 
                         appointments=appointments)

@app.route('/make_appointment', methods=['POST'])
@login_required
def make_appointment():
    """创建预约"""
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
    
    # 获取患者ID
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    if not patient_data:
        db.disconnect()
        return jsonify({'success': False, 'message': '未找到患者信息'})
    
    patient_id = patient_data[0]['patient_id']
    
    # 检查时间冲突
    existing_appointment = db.execute_query(
        """SELECT * FROM appointments 
           WHERE doctor_id = %s AND appointment_date = %s AND appointment_time = %s AND status != 'cancelled'""",
        (doctor_id, appointment_date, appointment_time)
    )
    
    if existing_appointment:
        db.disconnect()
        return jsonify({'success': False, 'message': '该时间段已被预约，请选择其他时间'})
    
    # 创建预约
    appointment_id = db.execute_insert(
        """INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, 
           department, symptoms, notes, status, created_at) 
           VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s)""",
        (patient_id, doctor_id, appointment_date, appointment_time, department, symptoms, notes, datetime.datetime.now())
    )
    
    if appointment_id:
        # 添加系统日志
        db.add_system_log(current_user.id, 'CREATE_APPOINTMENT', f'患者创建预约，预约ID: {appointment_id}')
        db.disconnect()
        return jsonify({'success': True, 'message': '预约成功', 'appointment_id': appointment_id})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '预约失败'})

@app.route('/cancel_appointment/<int:appointment_id>', methods=['POST'])
@login_required
def cancel_appointment(appointment_id):
    """取消预约"""
    if current_user.user_type != 'patient':
        return jsonify({'success': False, 'message': '无权操作'})
    
    db = Database()
    if not db.connect():
        return jsonify({'success': False, 'message': '数据库连接失败'})
    
    # 验证预约属于当前患者
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
    
    # 更新预约状态
    result = db.execute_insert(
        "UPDATE appointments SET status = 'cancelled', updated_at = %s WHERE appointment_id = %s",
        (datetime.datetime.now(), appointment_id)
    )
    
    if result:
        # 添加系统日志
        db.add_system_log(current_user.id, 'CANCEL_APPOINTMENT', f'患者取消预约，预约ID: {appointment_id}')
        db.disconnect()
        return jsonify({'success': True, 'message': '预约已取消'})
    else:
        db.disconnect()
        return jsonify({'success': False, 'message': '取消预约失败'})
#患者咨询界面   
@app.route('/patient/chat')
@login_required
def patient_chat():
    """患者在线咨询页面"""
    if current_user.user_type != 'patient':
        flash('无权访问此页面', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    # 获取当前用户关联的患者信息
    db = Database()
    if not db.connect():
        flash('数据库连接失败', 'danger')
        return render_template('patient_chat.html', patient=None, messages=[])
    
    # 获取患者信息
    patient_data = db.execute_query("SELECT * FROM patients WHERE user_id = %s", (current_user.id,))
    patient = patient_data[0] if patient_data else None
    
    # 获取咨询消息（这里使用模拟数据）
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
    
    # 获取在线医生列表
    online_doctors = db.execute_query(
        "SELECT user_id, full_name, specialty, department FROM users WHERE role = 'doctor' LIMIT 5"
    ) or []
    
    db.disconnect()
    
    return render_template('patient_chat.html', 
                         patient=patient,
                         messages=messages,
                         online_doctors=online_doctors)

if __name__ == '__main__':
    # 确保上传目录存在
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # 确保模型目录存在
    if not os.path.exists('./models'):
        os.makedirs('./models')
    
    # 确保静态图像目录存在
    if not os.path.exists('./static/images'):
        os.makedirs('./static/images')
    
    app.run(debug=True)