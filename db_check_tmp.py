# 临时数据库诊断脚本 - 表结构检查
import sys
sys.path.insert(0, r"C:\Users\张甜甜\Desktop\A Meta-Learning based Clinical Decision Support System for Rare Diseases")
from config import Config
import mysql.connector

c = mysql.connector.connect(
    host=Config.MYSQL_HOST,
    database=Config.MYSQL_DB,
    user=Config.MYSQL_USER,
    password=Config.MYSQL_PASSWORD
)
cur = c.cursor()

print("=== 数据库中的表 ===")
cur.execute("SHOW TABLES")
tables = [r[0] for r in cur.fetchall()]
print(tables)

for t in tables:
    print(f"\n=== 表 {t} 的结构 ===")
    cur.execute(f"SHOW CREATE TABLE {t}")
    row = cur.fetchone()
    print(row[1] if row else "EMPTY")
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    print("行数:", cur.fetchone()[0])

cur.close()
c.close()
