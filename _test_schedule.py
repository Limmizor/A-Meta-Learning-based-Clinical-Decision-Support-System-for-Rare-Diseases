# -*- coding: utf-8 -*-
"""临时测试脚本：模拟医生登录，验证 /doctor/schedule 页面与预约操作接口"""
import sys
sys.path.insert(0, r'C:\Users\张甜甜\Desktop\A Meta-Learning based Clinical Decision Support System for Rare Diseases')

import app as appmod
from database import Database

# 1) 查找一个医生账号
db = Database()
db.connect()
doctors = db.execute_query("SELECT user_id, username, role, full_name FROM users WHERE role='doctor' LIMIT 5") or []
print("DOCTORS:", doctors)
if not doctors:
    print("NO_DOCTOR_ACCOUNT")
    sys.exit(0)
doctor = doctors[0]
db.disconnect()

# 2) 用测试客户端登录
client = appmod.app.test_client()

# 通过 login POST 登录
resp = client.post('/login', data={
    'username': doctor['username'],
    'password': 'test'  # 密码未知，这里仅尝试；失败则用下面方式
}, follow_redirects=False)
print("LOGIN STATUS:", resp.status_code, "LOCATION:", resp.headers.get('Location'))

# 如果密码不对，直接往 session 里塞一个登录态
with client.session_transaction() as sess:
    sess['_user_id'] = str(doctor['user_id'])
    sess['_fresh'] = True

# 3) 请求 /doctor/schedule
resp = client.get('/doctor/schedule')
print("SCHEDULE STATUS:", resp.status_code)
html = resp.get_data(as_text=True)
print("SCHEDULE contains 今日日程:", '今日日程' in html)
print("SCHEDULE contains allAppointments:", 'allAppointments' in html)

# 4) 请求 /doctor/create_appointment（GET 应 405，POST 无参应返回 JSON 错误）
resp = client.post('/doctor/create_appointment', data={})
print("CREATE_EMPTY:", resp.status_code, resp.get_json())

# 5) 检查新路由存在
for rule in ['/doctor/appointment/1/confirm', '/doctor/appointment/1/complete', '/doctor/appointment/1/cancel']:
    r = client.post(rule)
    print(rule, "->", r.status_code, r.get_json())

print("DONE")
