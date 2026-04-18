import mysql.connector
from mysql.connector import Error
import datetime
from config import Config 

class Database:
    def __init__(self):
        # 使用Config中的配置，而不是硬编码的默认值
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

    # 疾病管理相关方法（字段名已统一为 treatment_options）
    def get_diseases(self):
        """获取所有疾病信息"""
        return self.execute_query("SELECT * FROM diseases ORDER BY disease_id DESC")

    def add_disease(self, name, description, symptoms, treatment_options):
        """添加新疾病"""
        return self.execute_insert(
            """INSERT INTO diseases (name, description, symptoms, treatment_options) 
               VALUES (%s, %s, %s, %s)""",
            (name, description, symptoms, treatment_options)
        )

    def update_disease(self, disease_id, name, description, symptoms, treatment_options):
        """更新疾病信息"""
        query = """UPDATE diseases SET name=%s, description=%s, symptoms=%s, 
                   treatment_options=%s, updated_at=%s WHERE disease_id=%s"""
        return self.execute_insert(
            query,
            (name, description, symptoms, treatment_options, datetime.datetime.now(), disease_id)
        )

    def delete_disease(self, disease_id):
        """删除疾病"""
        return self.execute_insert("DELETE FROM diseases WHERE disease_id=%s", (disease_id,))

    # 系统日志相关方法
    def get_system_logs(self, limit=100):
        """获取系统日志"""
        return self.execute_query(
            """SELECT l.*, u.username FROM system_logs l
               JOIN users u ON l.user_id = u.user_id
               ORDER BY log_time DESC LIMIT %s""",
            (limit,)
        )

    def add_system_log(self, user_id, action, details):
        """添加系统日志"""
        return self.execute_insert(
            """INSERT INTO system_logs (user_id, action, details, log_time)
               VALUES (%s, %s, %s, %s)""",
            (user_id, action, details, datetime.datetime.now())
        )

    # 患者管理相关方法
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

    # 医学影像相关方法
    def get_medical_images(self, patient_id):
        return self.execute_query(
            "SELECT * FROM medical_images WHERE patient_id = %s ORDER BY uploaded_at DESC",
            (patient_id,)
        )

    def add_medical_image(self, patient_id, filename, image_type, description):
        return self.execute_insert(
            """INSERT INTO medical_images (patient_id, filename, image_type, description, uploaded_at)
               VALUES (%s, %s, %s, %s, %s)""",
            (patient_id, filename, image_type, description, datetime.datetime.now())
        )

    # 诊断报告相关方法
    def get_diagnosis_reports(self, patient_id):
        return self.execute_query(
            """SELECT r.*, u.full_name as doctor_name FROM diagnosis_reports r
               JOIN users u ON r.doctor_id = u.user_id
               WHERE r.patient_id = %s ORDER BY created_at DESC""",
            (patient_id,)
        )

    def add_diagnosis_report(self, patient_id, doctor_id, clinical_notes, conclusion):
        return self.execute_insert(
            """INSERT INTO diagnosis_reports (patient_id, doctor_id, clinical_notes, conclusion, created_at)
               VALUES (%s, %s, %s, %s, %s)""",
            (patient_id, doctor_id, clinical_notes, conclusion, datetime.datetime.now())
        )

    # 疾病预测相关方法
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