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

    def execute_update(self, query, params=None):
        """执行 UPDATE/DELETE，返回受影响行数（用于判断是否成功）"""
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params or ())
            self.connection.commit()
            return cursor.rowcount
        except Error as e:
            print(f"更新错误: {e}")
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
                   treatment_options=%s WHERE disease_id=%s"""
        return self.execute_insert(
            query,
            (name, description, symptoms, treatment_options, disease_id)
        )

    def delete_disease(self, disease_id):
        return self.execute_insert("DELETE FROM diseases WHERE disease_id=%s", (disease_id,))

    # ---------- 系统日志 ----------
    def get_system_logs(self, limit=100):
        return self.execute_query(
            """SELECT l.*, u.username FROM system_logs l
               JOIN users u ON l.user_id = u.user_id
               ORDER BY l.created_at DESC LIMIT %s""",
            (limit,)
        )

    def add_system_log(self, user_id, action, details):
        return self.execute_insert(
            """INSERT INTO system_logs (user_id, action, details, created_at)
               VALUES (%s, %s, %s, %s)""",
            (user_id, action, details, datetime.datetime.now())
        )

    # ---------- 患者管理（含软删除 / 回收站）----------
    def get_patients(self, include_deleted=False):
        """获取患者列表，默认过滤已软删除患者"""
        if include_deleted:
            return self.execute_query("SELECT * FROM patients ORDER BY patient_id DESC")
        return self.execute_query(
            "SELECT * FROM patients WHERE is_deleted = 0 ORDER BY patient_id DESC"
        )

    def get_patient(self, patient_id, include_deleted=False):
        """获取单个患者，默认过滤已软删除患者"""
        if include_deleted:
            return self.execute_query(
                "SELECT * FROM patients WHERE patient_id = %s", (patient_id,)
            )
        return self.execute_query(
            "SELECT * FROM patients WHERE patient_id = %s AND is_deleted = 0", (patient_id,)
        )

    def get_deleted_patients(self):
        """获取已软删除的患者（回收站）"""
        return self.execute_query(
            "SELECT * FROM patients WHERE is_deleted = 1 ORDER BY updated_at DESC"
        )

    def soft_delete_patient(self, patient_id):
        """软删除患者：仅标记 is_deleted=1，关联数据保留可恢复"""
        return self.execute_update(
            "UPDATE patients SET is_deleted = 1 WHERE patient_id = %s", (patient_id,)
        )

    def restore_patient(self, patient_id):
        """恢复已软删除的患者"""
        return self.execute_update(
            "UPDATE patients SET is_deleted = 0 WHERE patient_id = %s", (patient_id,)
        )

    def purge_patient(self, patient_id):
        """彻底删除患者：依赖外键 ON DELETE CASCADE 自动清理全部关联数据（返回受影响行数）"""
        return self.execute_update("DELETE FROM patients WHERE patient_id = %s", (patient_id,))

    def add_patient(self, name, age, gender, contact_number, medical_history,
                    occupation=None, address=None, disease_type=None,
                    family_history=None, allergies=None):
        return self.execute_insert(
            """INSERT INTO patients (name, age, gender, contact_number, medical_history,
               occupation, address, disease_type, family_history, allergies, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (name, age, gender, contact_number, medical_history,
             occupation, address, disease_type, family_history, allergies,
             datetime.datetime.now())
        )

    # ---------- 医学影像 ----------
    def add_medical_image(self, patient_id, image_path, image_type, description):
        """插入医学影像记录（image_path 存储文件名）"""
        return self.execute_insert(
            """INSERT INTO medical_images (patient_id, image_path, image_type, description, upload_date)
               VALUES (%s, %s, %s, %s, %s)""",
            (patient_id, image_path, image_type, description, datetime.datetime.now())
        )

    def get_medical_images(self, patient_id):
        return self.execute_query(
            "SELECT * FROM medical_images WHERE patient_id = %s ORDER BY upload_date DESC",
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
                             lesion_area_ratio=None, distribution_range=None,
                             suggestions=None, differentials=None, conclusion_text=None):
        """添加诊断报告，同时保存量化指标（conclusion 写入 findings 列）"""
        return self.execute_insert(
            """INSERT INTO diagnosis_reports (patient_id, doctor_id, clinical_notes, findings,
               lesion_area_ratio, distribution_range, suggestions, differentials, conclusion, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (patient_id, doctor_id, clinical_notes, conclusion,
             lesion_area_ratio, distribution_range,
             suggestions, differentials, conclusion_text, datetime.datetime.now())
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
               WHERE p.report_id = %s ORDER BY `rank`""",
            (report_id,)
        )

    def add_disease_prediction(self, report_id, disease_id, confidence, rank):
        """rank 是 MySQL 保留字，需加反引号"""
        return self.execute_insert(
            """INSERT INTO disease_predictions (report_id, disease_id, confidence, `rank`)
               VALUES (%s, %s, %s, %s)""",
            (report_id, disease_id, confidence, rank)
        )

    # ---------- 医生复核（诊断报告 + AI 预测确认） ----------
    def get_all_reports(self):
        """获取全部诊断报告（含患者名/医生名），待复核优先，不含已软删除患者的报告"""
        return self.execute_query(
            """SELECT r.*, p.name AS patient_name, u.full_name AS doctor_name
               FROM diagnosis_reports r
               JOIN patients p ON r.patient_id = p.patient_id
               JOIN users u ON r.doctor_id = u.user_id
               WHERE p.is_deleted = 0
               ORDER BY FIELD(r.status, 'pending', 'completed', 'reviewed'), r.created_at DESC"""
        )

    def get_report_by_id(self, report_id):
        """按 ID 获取报告（含患者名/医生名）"""
        return self.execute_query(
            """SELECT r.*, p.name AS patient_name, u.full_name AS doctor_name
               FROM diagnosis_reports r
               JOIN patients p ON r.patient_id = p.patient_id
               JOIN users u ON r.doctor_id = u.user_id
               WHERE r.report_id = %s""",
            (report_id,)
        )

    def update_report_review(self, report_id, conclusion, status):
        """更新报告复核结果（诊断结论 + 状态 + 复核时间）"""
        return self.execute_update(
            """UPDATE diagnosis_reports SET findings = %s, status = %s, reviewed_at = %s
               WHERE report_id = %s""",
            (conclusion, status, datetime.datetime.now(), report_id)
        )

    def update_prediction_review(self, prediction_id, is_confirmed, confirmed_by, notes):
        """更新单条 AI 预测的复核结果"""
        return self.execute_update(
            """UPDATE disease_predictions SET is_confirmed = %s, confirmed_by = %s, notes = %s
               WHERE prediction_id = %s""",
            (1 if is_confirmed else 0, confirmed_by if is_confirmed else None, notes or None, prediction_id)
        )

    # ---------- 随访计划 ----------
    def create_followup_plan(self, patient_id, suggested_date, notes=None):
        """创建随访计划（notes 写入 reason 列）"""
        return self.execute_insert(
            """INSERT INTO followup_plans (patient_id, suggested_date, status, reason)
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

    def get_followup_plan(self, plan_id):
        """获取单个随访计划（用于权限校验）"""
        return self.execute_query(
            "SELECT * FROM followup_plans WHERE plan_id = %s", (plan_id,)
        )

    def get_all_followup_plans(self, status=None):
        """医生端：获取全部患者的随访计划（含患者姓名）"""
        if status:
            return self.execute_query(
                """SELECT f.*, p.name AS patient_name, p.age AS patient_age, p.gender AS patient_gender
                   FROM followup_plans f
                   JOIN patients p ON f.patient_id = p.patient_id
                   WHERE f.status = %s AND p.is_deleted = 0
                   ORDER BY f.suggested_date ASC""",
                (status,)
            )
        return self.execute_query(
            """SELECT f.*, p.name AS patient_name, p.age AS patient_age, p.gender AS patient_gender
               FROM followup_plans f
               JOIN patients p ON f.patient_id = p.patient_id
               WHERE p.is_deleted = 0
               ORDER BY FIELD(f.status, 'pending', 'completed', 'cancelled'), f.suggested_date ASC"""
        )

    def update_followup_status(self, plan_id, status):
        """更新随访计划状态"""
        return self.execute_insert(
            "UPDATE followup_plans SET status = %s WHERE plan_id = %s",
            (status, plan_id)
        )

    def delete_followup_plan(self, plan_id):
        """删除随访计划"""
        return self.execute_insert("DELETE FROM followup_plans WHERE plan_id = %s", (plan_id,))

    # ---------- 患者健康日志（health_logs）----------
    def add_health_log(self, patient_id, log_date, symptom_score=None, medication=None, notes=None):
        """新增患者健康日志（症状评分、用药、日常记录）"""
        return self.execute_insert(
            """INSERT INTO health_logs (patient_id, log_date, symptom_score, medication, notes)
               VALUES (%s, %s, %s, %s, %s)""",
            (patient_id, log_date, symptom_score, medication, notes)
        )

    def get_health_logs(self, patient_id, limit=None):
        """获取患者的健康日志，按日期倒序"""
        if limit:
            return self.execute_query(
                "SELECT * FROM health_logs WHERE patient_id = %s ORDER BY log_date DESC, log_id DESC LIMIT %s",
                (patient_id, limit)
            )
        return self.execute_query(
            "SELECT * FROM health_logs WHERE patient_id = %s ORDER BY log_date DESC, log_id DESC",
            (patient_id,)
        )

    def get_health_log(self, patient_id, log_id):
        """获取单条健康日志（用于权限校验）"""
        return self.execute_query(
            "SELECT * FROM health_logs WHERE patient_id = %s AND log_id = %s",
            (patient_id, log_id)
        )

    def update_health_log(self, log_id, log_date, symptom_score=None, medication=None, notes=None):
        """更新健康日志"""
        return self.execute_update(
            """UPDATE health_logs SET log_date = %s, symptom_score = %s, medication = %s, notes = %s
               WHERE log_id = %s""",
            (log_date, symptom_score, medication, notes, log_id)
        )

    def delete_health_log(self, log_id):
        """删除健康日志"""
        return self.execute_insert("DELETE FROM health_logs WHERE log_id = %s", (log_id,))

    def get_health_log_trend(self, patient_id):
        """获取患者健康日志的症状评分趋势（供趋势分析使用）"""
        return self.execute_query(
            """SELECT log_date, symptom_score, medication, notes
               FROM health_logs
               WHERE patient_id = %s AND symptom_score IS NOT NULL
               ORDER BY log_date ASC""",
            (patient_id,)
        )

    # ---------- 消息通知 ----------
    def add_notification(self, user_id, title, content=None, notification_type='system', link=None):
        """创建一条消息通知"""
        return self.execute_insert(
            """INSERT INTO notifications (user_id, title, content, notification_type, link)
               VALUES (%s, %s, %s, %s, %s)""",
            (user_id, title, content, notification_type, link)
        )

    def get_notifications(self, user_id, limit=20):
        """获取用户的最近通知列表"""
        return self.execute_query(
            """SELECT * FROM notifications WHERE user_id = %s 
               ORDER BY created_at DESC, notification_id DESC LIMIT %s""",
            (user_id, limit)
        )

    def get_unread_notification_count(self, user_id):
        """获取用户未读通知数量"""
        result = self.execute_query(
            "SELECT COUNT(*) as count FROM notifications WHERE user_id = %s AND is_read = 0",
            (user_id,)
        )
        return result[0]['count'] if result else 0

    def mark_notification_read(self, notification_id, user_id):
        """标记单条通知为已读"""
        return self.execute_insert(
            "UPDATE notifications SET is_read = 1 WHERE notification_id = %s AND user_id = %s",
            (notification_id, user_id)
        )

    def mark_all_notifications_read(self, user_id):
        """标记用户全部通知为已读"""
        return self.execute_insert(
            "UPDATE notifications SET is_read = 1 WHERE user_id = %s AND is_read = 0",
            (user_id,)
        )

    # ---------- 在线咨询 / 聊天 ----------
    def find_conversation(self, patient_id, doctor_id):
        """查找患者与医生之间进行中的会话"""
        return self.execute_query(
            """SELECT * FROM chat_conversations 
               WHERE patient_id = %s AND doctor_id = %s AND status = 'open' LIMIT 1""",
            (patient_id, doctor_id)
        )

    def create_conversation(self, patient_id, doctor_id):
        """创建患者与医生的新会话"""
        return self.execute_insert(
            """INSERT INTO chat_conversations (patient_id, doctor_id, status) 
               VALUES (%s, %s, 'open')""",
            (patient_id, doctor_id)
        )

    def get_conversation(self, conversation_id):
        """获取单个会话详情"""
        return self.execute_query(
            "SELECT * FROM chat_conversations WHERE conversation_id = %s",
            (conversation_id,)
        )

    def get_user_conversations(self, user_id, role):
        """获取某个用户（患者或医生）的全部会话，附带对方信息和最后消息"""
        if role == 'doctor':
            return self.execute_query(
                """SELECT c.*, u.full_name as peer_name, u.user_id as peer_id, 
                          u.role as peer_role
                   FROM chat_conversations c
                   JOIN users u ON c.patient_id = u.user_id
                   WHERE c.doctor_id = %s
                   ORDER BY c.updated_at DESC""",
                (user_id,)
            )
        else:
            return self.execute_query(
                """SELECT c.*, u.full_name as peer_name, u.user_id as peer_id,
                          u.role as peer_role
                   FROM chat_conversations c
                   JOIN users u ON c.doctor_id = u.user_id
                   WHERE c.patient_id = %s
                   ORDER BY c.updated_at DESC""",
                (user_id,)
            )

    def get_conversation_messages(self, conversation_id, limit=100):
        """获取会话的消息记录（从旧到新）"""
        return self.execute_query(
            """SELECT * FROM (
                   SELECT * FROM chat_messages WHERE conversation_id = %s 
                   ORDER BY message_id DESC LIMIT %s
               ) sub ORDER BY message_id ASC""",
            (conversation_id, limit)
        )

    def get_conversation_unread_count(self, conversation_id, receiver_id):
        """获取会话中指定接收者的未读消息数"""
        result = self.execute_query(
            """SELECT COUNT(*) as count FROM chat_messages 
               WHERE conversation_id = %s AND receiver_id = %s AND is_read = 0""",
            (conversation_id, receiver_id)
        )
        return result[0]['count'] if result else 0

    def get_total_unread_chat_count(self, user_id):
        """获取用户在所有会话中的未读消息总数"""
        result = self.execute_query(
            """SELECT COUNT(*) as count FROM chat_messages 
               WHERE receiver_id = %s AND is_read = 0""",
            (user_id,)
        )
        return result[0]['count'] if result else 0

    def add_chat_message(self, conversation_id, sender_id, receiver_id, content):
        """发送一条聊天消息，并更新会话的最后消息"""
        message_id = self.execute_insert(
            """INSERT INTO chat_messages (conversation_id, sender_id, receiver_id, content)
               VALUES (%s, %s, %s, %s)""",
            (conversation_id, sender_id, receiver_id, content)
        )
        if message_id:
            self.execute_insert(
                "UPDATE chat_conversations SET last_message = %s WHERE conversation_id = %s",
                (content[:200], conversation_id)
            )
        return message_id

    def mark_conversation_read(self, conversation_id, user_id):
        """将会话中发给指定用户的消息标记为已读"""
        return self.execute_insert(
            """UPDATE chat_messages SET is_read = 1 
               WHERE conversation_id = %s AND receiver_id = %s AND is_read = 0""",
            (conversation_id, user_id)
        )

    def close_conversation(self, conversation_id):
        """结束会话"""
        return self.execute_insert(
            "UPDATE chat_conversations SET status = 'closed' WHERE conversation_id = %s",
            (conversation_id,)
        )
