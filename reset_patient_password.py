from werkzeug.security import generate_password_hash
import mysql.connector
from config import Config

def reset_patient_password():
    # 连接数据库
    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB
    )
    cursor = conn.cursor()

    # 获取所有患者账号（role = 'patient'）
    cursor.execute("SELECT user_id, username FROM users WHERE role = 'patient'")
    patients = cursor.fetchall()

    if not patients:
        print("❌ 当前数据库中没有患者账号。")
        print("你可以先运行 init_db.py 创建默认患者账号 patient1 / 123456")
        cursor.close()
        conn.close()
        return

    print("现有的患者账号：")
    for idx, (uid, uname) in enumerate(patients, 1):
        print(f"  {idx}. {uname} (user_id={uid})")

    # 选择或输入用户名
    choice = input("\n请输入要重置密码的患者用户名（或输入序号）: ").strip()
    target_username = None

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(patients):
            target_username = patients[idx][1]
        else:
            print("❌ 序号无效")
            cursor.close()
            conn.close()
            return
    else:
        target_username = choice

    # 验证用户是否存在且为患者
    cursor.execute("SELECT user_id FROM users WHERE username = %s AND role = 'patient'", (target_username,))
    user = cursor.fetchone()
    if not user:
        print(f"❌ 患者用户 '{target_username}' 不存在，请检查用户名")
        cursor.close()
        conn.close()
        return

    # 输入新密码
    new_password = input(f"请输入患者 {target_username} 的新密码: ").strip()
    if not new_password:
        print("❌ 密码不能为空")
        cursor.close()
        conn.close()
        return

    # 生成哈希并更新
    new_hash = generate_password_hash(new_password)
    cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (new_hash, target_username))
    conn.commit()

    if cursor.rowcount > 0:
        print(f"✅ 患者 '{target_username}' 的密码已成功重置为: {new_password}")
    else:
        print("❌ 更新失败，请检查数据库连接")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    reset_patient_password()