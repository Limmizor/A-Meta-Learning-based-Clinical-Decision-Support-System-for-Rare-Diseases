"""数据库迁移脚本：为现有表补充软删除字段等结构性变更（幂等，可重复执行）"""
import mysql.connector
from mysql.connector import Error
from config import Config


def column_exists(cursor, table, column):
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s",
        (table, column)
    )
    return cursor.fetchone()[0] > 0


def run_migrations():
    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        database=Config.MYSQL_DB,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD
    )
    cursor = conn.cursor()
    applied = []

    # 1. patients 表增加软删除标记（回收站支持）
    if not column_exists(cursor, 'patients', 'is_deleted'):
        cursor.execute(
            "ALTER TABLE patients ADD COLUMN is_deleted TINYINT(1) NOT NULL DEFAULT 0 "
            "COMMENT '软删除标记：0正常 1已删除'"
        )
        applied.append("patients.is_deleted")

    if applied:
        conn.commit()
        print("已应用的迁移:", ", ".join(applied))
    else:
        print("无需迁移，所有字段已存在。")

    cursor.close()
    conn.close()


if __name__ == '__main__':
    try:
        run_migrations()
    except Error as e:
        print(f"数据库迁移失败: {e}")
