# 重置用户密码的脚本
from werkzeug.security import generate_password_hash
import mysql.connector
from config import Config

username = input("请输入要重置密码的用户名: ")
new_password = input("请输入新密码: ")

conn = mysql.connector.connect(
    host=Config.MYSQL_HOST,
    user=Config.MYSQL_USER,
    password=Config.MYSQL_PASSWORD,
    database=Config.MYSQL_DB
)
cursor = conn.cursor()

new_hash = generate_password_hash(new_password)
cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (new_hash, username))
if cursor.rowcount > 0:
    print(f"用户 {username} 的密码已重置为 {new_password}")
else:
    print(f"用户 {username} 不存在")
conn.commit()
cursor.close()
conn.close()