# -*- coding: utf-8 -*-
"""临时测试脚本：端到端验证医生端预约流程"""
import sys
sys.path.insert(0, r'C:\Users\张甜甜\Desktop\A Meta-Learning based Clinical Decision Support System for Rare Diseases')

import app as appmod
from database import Database

db = Database()
db.connect()
# 找医生和患者
doctor = (db.execute_query("SELECT user_id, username FROM users WHERE role='doctor' AND password_hash != '' LIMIT 1") or
          db.execute_query("SELECT user_id, username FROM users WHERE role='doctor' LIMIT 1"))[0]
patient = (db.execute_query("SELECT * FROM patients LIMIT 1") or [None])[0]
print("DOCTOR:", doctor, "PATIENT:", patient and patient['patient_id'])
if not patient:
    print("NO_PATIENT")
    sys.exit(0)
patient_id = patient['patient_id']
# 清理可能残留的同医生测试预约
db.execute_insert("DELETE FROM appointments WHERE doctor_id=%s AND notes='AUTOTEST'", (doctor['user_id'],))
db.disconnect()

client = appmod.app.test_client()
with client.session_transaction() as sess:
    sess['_user_id'] = str(doctor['user_id'])
    sess['_fresh'] = True

# 1) 医生为患者新建预约
r = client.post('/doctor/create_appointment', data={
    'patient_id': str(patient_id),
    'appointment_date': '2026-08-20',
    'appointment_time': '10:30',
    'department': '罕见病科',
    'symptoms': '咳嗽、气短',
    'notes': 'AUTOTEST'
})
print("CREATE:", r.status_code, r.get_json())
aid = r.get_json().get('appointment_id')

# 2) 确认预约
r = client.post(f'/doctor/appointment/{aid}/confirm')
print("CONFIRM:", r.status_code, r.get_json())

# 3) 完成预约
r = client.post(f'/doctor/appointment/{aid}/complete')
print("COMPLETE:", r.status_code, r.get_json())

# 4) 再建一条并取消
r = client.post('/doctor/create_appointment', data={
    'patient_id': str(patient_id),
    'appointment_date': '2026-08-21',
    'appointment_time': '09:00',
    'department': '罕见病科',
    'symptoms': '复查',
    'notes': 'AUTOTEST'
})
aid2 = r.get_json().get('appointment_id')
r = client.post(f'/doctor/appointment/{aid2}/cancel')
print("CANCEL:", r.status_code, r.get_json())

# 5) 检查日程页包含新建预约
html = client.get('/doctor/schedule').get_data(as_text=True)
print("SCHEDULE shows patient:", patient['name'] in html, "| has create modal:", 'appointmentModal' in html)

# 6) 越权校验：用非该医生的账号操作应失败
with client.session_transaction() as sess:
    sess['_user_id'] = '999999'
    sess['_fresh'] = True
r = client.post(f'/doctor/appointment/{aid}/confirm')
print("UNAUTHORIZED (no such doctor):", r.status_code, r.get_json())
print("DONE")
