# 肺影智诊 · 部署说明

## 〇、已部署实例（2026-08-15）

- 公网地址：http://114.55.208.128 （阿里云 ECS，华东1杭州）
- 规格：2 vCPU / 4 GiB / Ubuntu 22.04（已从免费 2G 实例升级）
- 架构：nginx(80) → waitress(127.0.0.1:5000) → Flask；MySQL 本机；swap 4G
- 服务：pf、nginx、mysql 均已设开机自启
- 演示账号：医生、患者各一个（账号凭据不随仓库公开，见本地《部署验收报告.docx》或另行提供）
- 云上单次 AI 诊断实测约 75–80 秒（2 核 CPU）；本机（8G 内存）约 30 秒
- ⚠️ 部署过程使用的 SSH root 密码与 MySQL root 密码已在本机对话中出现，正式对外使用前建议修改

## 一、当前部署状态（本机 Windows 生产模式）

已完成本地生产部署：

- 生产服务器：waitress（WSGI），监听 `0.0.0.0:5000`，8 线程
- 数据库：本机 MySQL80（库名 `rare_disease_diagnosis`），服务已设为开机自启
- 模型文件：`models/best_maml_fold1.pth` + `models/ipf_support_set.pt`（本地就位）
- `SECRET_KEY`：已替换为随机强密钥
- 访问地址：
  - 本机：http://127.0.0.1:5000
  - 局域网：http://192.168.2.5:5000 （以实际 `ipconfig` 为准）

## 二、启动 / 停止

- 启动：双击 `start_prod.bat`（自动清理 5000 端口旧实例 → 启动 waitress → 打开浏览器）
- 停止：双击 `stop_prod.bat`，或关闭名为 `PF-Server-Prod` 的最小化窗口
- 日志：`prod_out.log`（标准输出）、`prod_err.log`（错误）

## 三、局域网访问（Windows 防火墙）

当前脚本已能本机/局域网 IP 访问；若其他设备访问超时，需以**管理员身份**在 PowerShell 执行一次：

```powershell
New-NetFirewallRule -DisplayName "肺影智诊-5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Private
```

## 四、换一台 Windows 机器部署

1. 安装 Python 3.12（勾选 Add to PATH）与 MySQL 8.0。
2. 拷贝代码目录（排除 `.git`、`.venv`、`data`、`static/uploads|previews|thumbnails|gradcam`）。
3. **拷贝模型文件**（不在 git 仓库）：`models/best_maml_fold1.pth`、`models/ipf_support_set.pt`。
4. 导出并导入数据库：

```bash
mysqldump -u root -p --default-character-set=utf8mb4 rare_disease_diagnosis > deploy_schema.sql
# 目标机：
mysql -u root -p rare_disease_diagnosis < deploy_schema.sql
```

5. 安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install waitress
```

6. 修改 `config.py` 的 `MYSQL_PASSWORD`；执行 `python db_migrations.py`。
7. 双击 `start_prod.bat` 启动，注册医生账号（`/register` 可选医生/患者角色）。

## 五、阿里云 ECS 部署（Linux）

### 需要准备的账号信息

- 阿里云账号已实名认证，并同意开通 ECS（涉及费用）。
- 建议实例：**轻量应用服务器或 ECS，2核4G**（CPU 推理约 30 秒，够用），系统 **Ubuntu 22.04/24.04**，带宽 3M 起或按量。
- 安全组/防火墙放行：**22（SSH）、80/443（如需域名）、5000（可选直连）**。
- 提供给部署方的连接信息：**公网 IP + SSH 端口 + 用户名 + 密码或密钥**。

### Linux 部署步骤

```bash
# 1. 基础环境
apt update && apt install -y python3.12 python3.12-venv mysql-server nginx fonts-noto-cjk

# 2. 上传代码与模型文件（scp 或宝塔），目录假设 /opt/pf
cd /opt/pf
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install waitress

# 3. 数据库
mysql -e "CREATE DATABASE rare_disease_diagnosis DEFAULT CHARACTER SET utf8mb4;"
mysql -u root -p rare_disease_diagnosis < deploy_schema.sql

# 4. 修改 config.py（MYSQL_PASSWORD、SECRET_KEY），执行
.venv/bin/python db_migrations.py
```

⚠️ **中文字体**：`pdf_report.py` 默认只注册 Windows 字体（`C:\Windows\Fonts`），Linux 上必须把 `_register_fonts()` 中的字体路径改为 Noto Sans CJK（`/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` 等），否则导出的 PDF 中文会乱码。

```bash
# 5. systemd 服务 /etc/systemd/system/pf.service
[Unit]
Description=肺影智诊
After=network.target mysql.service

[Service]
WorkingDirectory=/opt/pf
ExecStart=/opt/pf/.venv/bin/python -m waitress --listen=127.0.0.1:5000 --threads=8 app:app
Restart=always

[Install]
WantedBy=multi-user.target

systemctl enable --now pf

# 6. nginx 反向代理 /etc/nginx/sites-available/pf
server {
    listen 80;
    server_name _;
    client_max_body_size 128m;
    location / { proxy_pass http://127.0.0.1:5000; proxy_set_header Host $host; proxy_read_timeout 300s; }
}
```

## 六、演示前验收清单

- 医生登录正常；准备演示账号。
- 提前上传 20–30 张 CT 切片到患者档案（首次诊断 CPU 约 30 秒，别现场等）。
- 完整跑一次 AI 诊断：主诊断、鉴别诊断、热力图、切片统计正常。
- 导出 PDF：中文无乱码、版式完整。
- 患者端"我的报告"、医生端"报告复核"疾病名正常。
- 刷新不掉登录；`SECRET_KEY` 已修改。
- 数据与模型文件单独备份（`mysqldump` + 拷贝 `models/*.pth`）。
