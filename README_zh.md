# MMKG Project

这是一个多模态商品知识图谱项目，当前已经包含模型训练/推理代码、FastAPI 后端、用于 3D 知识图谱查询的 Flask 服务，以及 Vite React 前端。

更完整的交接说明见 [docs/HANDOVER.md](docs/HANDOVER.md)。

## 运行组件

- FastAPI 后端：`http://127.0.0.1:8000`
- Flask 知识图谱查询服务：`http://127.0.0.1:5000`
- Vite React 前端：`http://localhost:3000`

## 项目结构

```text
mmkg-project/
  backend/      FastAPI 应用和 Flask KG 查询入口
  frontend/     React + Vite 前端应用
  kg/           OpenBG-IMG 知识图谱数据转换脚本
  ml/           训练、推理、配置和本地模型产物
  data/         数据集、处理后的 KG 文件和向量缓存
  docs/         交接文档、实验协议和项目说明
  scripts/      一键安装/启动脚本
```

## 必要本地数据

OpenBG-IMG 原始数据应放在：

```text
data/datasets/openbg_img/raw/
```

KG 可视化还需要：

```text
data/datasets/openbg_img/processed/data.csv
data/datasets/openbg_img/processed/metadata.json
```

这两个文件可以通过 `kg/` 下的脚本生成：

```bash
cd kg
python convert_openbg.py
python generate_metadata.py
```

训练好的模型产物默认放在：

```text
ml/artifacts/production_models/
```

模型 checkpoint 文件较大，已经被 Git 忽略。Attribute Completion 后端会优先检查 `ml/artifacts/production_models/gate+residual/`，如果没有可用 production run，则回退到 `ml/artifacts/outputs/openbg_img_gated_vec_res_rel/`。

## 一键安装

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

macOS/Linux：

```bash
bash scripts/install.sh
```

安装脚本会安装 Python 依赖、安装前端依赖，并在原始 OpenBG-IMG 数据存在时自动生成 KG processed 数据。

## 启动项目

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```

macOS/Linux：

```bash
bash scripts/start-dev.sh
```

然后打开：

```text
http://localhost:3000
```

在 macOS/Linux 上，日志和 pid 文件会写入 `.runtime_logs/`。停止后台服务：

```bash
bash scripts/stop-dev.sh
```

## 手动启动

如果需要手动控制，可以打开三个终端。

FastAPI：

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Flask KG 服务：

```bash
cd backend
python flask_app.py
```

前端：

```bash
cd frontend
npm install
npm run dev
```

## 常用文档

- [docs/HANDOVER.md](docs/HANDOVER.md)：完整安装、启动、数据、模型和排错说明
- [kg/README.md](kg/README.md)：KG processed 数据生成说明
- [backend/README.md](backend/README.md)：后端服务和接口说明
- [frontend/README.md](frontend/README.md)：前端开发说明
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md)：实验与复现协议
