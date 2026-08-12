import os

class Config:
    # 数据库配置
    MYSQL_HOST = 'localhost'
    MYSQL_USER = 'root'
    MYSQL_PASSWORD = '981812'
    MYSQL_DB = 'rare_disease_diagnosis'
    
    # 应用配置
    SECRET_KEY = 'your_secret_key_here'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64MB 文件上传限制（支持30张DICOM序列一次上传）
    
    # 确保上传目录存在
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
