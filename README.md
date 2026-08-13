# 肺影智诊 · 肺纤维化临床决策支持系统

基于 Flask 的肺纤维化临床决策支持系统，集成 MAML 元迁移学习模型，
对肺纤维化 CT 影像进行患者级分型辅助识别（主要诊断 + 鉴别诊断参考），
并提供 AI 辅助诊断、Grad-CAM 可视化与规范化 PDF 报告导出。

## 一、环境要求

- Windows + Python 3.12
- MySQL 5.7 / 8.0（本地数据库）
- 依赖安装：

```bash
pip install -r requirements.txt
```

> 训练 MAML 模型额外需要 `learn2learn`（已写入 requirements），仅推理不需要。

## 二、数据库配置

1. 启动 MySQL，创建数据库：

```sql
CREATE DATABASE rare_disease_diagnosis DEFAULT CHARACTER SET utf8mb4;
```

2. 在 `config.py` 中核对数据库账号密码（默认 root / 981812，可自行修改）。
3. 若表结构有变更，可执行迁移脚本：

```bash
python db_migrations.py
```

## 三、模型与数据文件

| 文件 | 说明 |
| --- | --- |
| `models/best_maml_fold1.pth` | MAML 训练好的 ResNet-18 肺纤维化分型识别模型（第 1 折最优） |
| `models/ipf_support_set.pt` | 推理用支持集（第 1 折训练集每类 2 名患者 × 8 张切片），用于 MAML 快速适应 |
| `data/osic-pulmonary-fibrosis-progression/` | OSIC 肺纤维化数据集（train.csv + train/ 患者 DICOM） |

支持集由 `prep_support_set.py` 生成；更换模型后如需重新生成：

```bash
python prep_support_set.py
```

## 四、启动服务

方式一（推荐）：双击 `start_server.bat`，脚本会自动清理端口 5000 上的旧实例并启动。

方式二（手动）：

```bash
python app.py
```

访问地址：**http://127.0.0.1:5000**

医生登录后进入「AI辅助诊断」：选择患者 → 上传 DICOM / JPG / PNG CT 切片（建议 30 张）→
自动完成支持集适应与患者级综合判读 → 查看诊断结果 → 「导出PDF报告」。

## 五、模型训练（可选）

```bash
python train_model.py
```

脚本使用 OSIC 真实数据执行 5 折患者级分层交叉验证的 MAML 训练，训练产物保存到 `models/`。
训练需 GPU 与 `learn2learn`，耗时较长，日常使用无需重新训练。

## 六、常见问题

- **数据库连接失败**：确认 MySQL 已启动、`config.py` 账号密码正确；同一时间只启动一个服务实例。
- **端口被占用**：使用 `start_server.bat` 启动，或手动结束占用 5000 端口的进程后再启动。
- **DICOM 无法读取**：确认已安装 `pydicom`、`pylibjpeg`、`pylibjpeg-libjpeg`。
