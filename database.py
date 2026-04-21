import mysql.connector
from mysql.connector import Error
import datetime
from config import Config 

class Database:
    def __init__(self):
        self.config = {
            'host': Config.MYSQL_HOST,
            'database': Config.MYSQL_DB,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD
        }
        self.connection = None

    def connect(self):
        try:
            self.connection = mysql.connector.connect(**self.config)
            return self.connection.is_connected()
        except Error as e:
            print(f"数据库连接错误: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def execute_query(self, query, params=None):
        cursor = self.connection.cursor(dictionary=True)
        try:
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            return result
        except Error as e:
            print(f"查询错误: {e}")
            return None
        finally:
            cursor.close()

    def execute_insert(self, query, params=None):
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params or ())
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"插入错误: {e}")
            self.connection.rollback()
            return None
        finally:
            cursor.close()

    # ---------- 疾病管理 ----------
    def get_diseases(self):
        return self.execute_query("SELECT * FROM diseases ORDER BY disease_id DESC")

    def add_disease(self, name, description, symptoms, treatment_options):
        return self.execute_insert(
            """INSERT INTO diseases (name, description, symptoms, treatment_options) 
               VALUES (%s, %s, %s, %s)""",
            (name, description, symptoms, treatment_options)
        )

    def update_disease(self, disease_id, name, description, symptoms, treatment_options):
        query = """UPDATE diseases SET name=%s, description=%s, symptoms=%s, 
                   treatment_options=%s, updated_at=%s WHERE disease_id=%s"""
        return self.execute_insert(
            query,
            (name, description, symptoms, treatment_options, datetime.datetime.now(), disease_id)
        )

    def delete_disease(self, disease_id):
        return self.execute_insert("DELETE FROM diseases WHERE disease_id=%s", (disease_id,))

    # ---------- 系统日志 ----------
    def get_system_logs(self, limit=100):
        return self.execute_query(
            """SELECT l.*, u.username FROM system_logs l
               JOIN users u ON l.user_id = u.user_id
               ORDER BY log_time DESC LIMIT %s""",
            (limit,)
        )

    def add_system_log(self, user_id, action, details):
        return self.execute_insert(
            """INSERT INTO system_logs (user_id, action, details, log_time)
               VALUES (%s, %s, %s, %s)""",
            (user_id, action, details, datetime.datetime.now())
        )

    # ---------- 患者管理 ----------
    def get_patients(self):
        return self.execute_query("SELECT * FROM patients ORDER BY patient_id DESC")

    def get_patient(self, patient_id):
        return self.execute_query("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))

    def add_patient(self, name, age, gender, contact_number, medical_history):
        return self.execute_insert(
            """INSERT INTO patients (name, age, gender, contact_number, medical_history, created_at)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (name, age, gender, contact_number, medical_history, datetime.datetime.now())
        )

    # ---------- 医学影像 ----------
    def add_medical_image(self, patient_id, image_path, image_type, description):
        """插入医学影像记录（image_path 存储文件名）"""
        return self.execute_insert(
            """INSERT INTO medical_images (patient_id, image_path, image_type, description, uploaded_at)
               VALUES (%s, %s, %s, %s, %s)""",
            (patient_id, image_path, image_type, description, datetime.datetime.now())
        )

    def get_medical_images(self, patient_id):
        return self.execute_query(
            "SELECT * FROM medical_images WHERE patient_id = %s ORDER BY uploaded_at DESC",
            (patient_id,)
        )

    # ---------- 诊断报告（扩展量化指标）----------
    def get_diagnosis_reports(self, patient_id):
        return self.execute_query(
            """SELECT r.*, u.full_name as doctor_name FROM diagnosis_reports r
               JOIN users u ON r.doctor_id = u.user_id
               WHERE r.patient_id = %s ORDER BY created_at DESC""",
            (patient_id,)
        )

    def add_diagnosis_report(self, patient_id, doctor_id, clinical_notes, conclusion,
                             lesion_area_ratio=None, distribution_range=None):
        """添加诊断报告，同时保存量化指标"""
        return self.execute_insert(
            """INSERT INTO diagnosis_reports (patient_id, doctor_id, clinical_notes, conclusion, 
               lesion_area_ratio, distribution_range, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (patient_id, doctor_id, clinical_notes, conclusion,
             lesion_area_ratio, distribution_range, datetime.datetime.now())
        )

    def get_patient_trend_data(self, patient_id):
        """获取患者所有诊断报告的趋势数据（日期、病灶面积占比、分布范围）"""
        return self.execute_query(
            """SELECT created_at, lesion_area_ratio, distribution_range 
               FROM diagnosis_reports 
               WHERE patient_id = %s AND lesion_area_ratio IS NOT NULL
               ORDER BY created_at ASC""",
            (patient_id,)
        )

    # ---------- 疾病预测 ----------
    def get_disease_predictions(self, report_id):
        return self.execute_query(
            """SELECT p.*, d.name as disease_name FROM disease_predictions p
               JOIN diseases d ON p.disease_id = d.disease_id
               WHERE p.report_id = %s ORDER BY rank""",
            (report_id,)
        )

    def add_disease_prediction(self, report_id, disease_id, confidence, rank):
        return self.execute_insert(
            """INSERT INTO disease_predictions (report_id, disease_id, confidence, rank)
               VALUES (%s, %s, %s, %s)""",
            (report_id, disease_id, confidence, rank)
        )

    # ---------- 随访计划 ----------
    def create_followup_plan(self, patient_id, suggested_date, notes=None):
        """创建随访计划"""
        return self.execute_insert(
            """INSERT INTO followup_plans (patient_id, suggested_date, status, notes)
               VALUES (%s, %s, 'pending', %s)""",
            (patient_id, suggested_date, notes)
        )

    def get_followup_plans(self, patient_id, status=None):
        """获取患者的随访计划，可按状态筛选"""
        if status:
            return self.execute_query(
                "SELECT * FROM followup_plans WHERE patient_id = %s AND status = %s ORDER BY suggested_date ASC",
                (patient_id, status)
            )
        else:
            return self.execute_query(
                "SELECT * FROM followup_plans WHERE patient_id = %s ORDER BY suggested_date ASC",
                (patient_id,)
            )

    def update_followup_status(self, plan_id, status):
        """更新随访计划状态"""
        return self.execute_insert(
            "UPDATE followup_plans SET status = %s, updated_at = %s WHERE plan_id = %s",
            (status, datetime.datetime.now(), plan_id)
        )

    def delete_followup_plan(self, plan_id):
        """删除随访计划"""
        return self.execute_insert("DELETE FROM followup_plans WHERE plan_id = %s", (plan_id,))