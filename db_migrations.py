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

    # 2. 补齐肺纤维化分型目录（数据补齐，幂等；不修改表结构）
    catalog = [
        ('特发性肺纤维化(IPF)', 'J84.1',
         '最常见的特发性间质性肺炎，呈进行性肺纤维化，影像以双肺底胸膜下网格影及蜂窝影为主。',
         '干咳、活动后呼吸困难、杵状指',
         '抗纤维化药物治疗（吡非尼酮/尼达尼布），肺康复与规律随访管理'),
        ('隐源性机化性肺炎(COP)', 'J84.116',
         '以肺泡管及肺泡腔内机化性肉芽组织为特征的间质性肺病，影像常见游走性实变影或磨玻璃影。',
         '发热、干咳、进行性呼吸困难',
         '糖皮质激素治疗，多数患者反应良好'),
        ('药物性肺纤维化', 'J70.4',
         '由药物（如胺碘酮、博来霉素等）引起的间质性肺炎及肺纤维化。',
         '干咳、呼吸困难，与可疑用药史相关',
         '停用可疑药物，结合糖皮质激素等对症治疗'),
        ('其他间质性肺病', 'J84.9',
         '未归入上述类别的其他间质性肺病，需结合HRCT表现与临床资料综合鉴别。',
         '咳嗽、呼吸困难、影像间质改变',
         '结合病因治疗，定期复查肺功能与影像'),
    ]
    for name, icd, desc, symptoms, treatment in catalog:
        cursor.execute("SELECT COUNT(*) FROM diseases WHERE name = %s", (name,))
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO diseases (name, omim_id, description, symptoms, treatment_options) "
                "VALUES (%s, %s, %s, %s, %s)",
                (name, icd, desc, symptoms, treatment))
            applied.append(f"diseases:{name}")

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
