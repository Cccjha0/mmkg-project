# MMKG Project

这是一个面向多模态知识图谱项目的总仓库。当前已经完成的是 `ml` 下的训练框架，`backend`、`frontend`、`docs` 和 `ml/inference` 现在是为后续开发准备的标准目录。

## 新仓库结构

```text
mmkg-project/
  ml/
    training/
      src/
      scripts/
    inference/
    artifacts/
      outputs/
    configs/
  backend/
  frontend/
  data/
    datasets/
    cache/
  docs/
```

## 目录职责

- `ml/training`：训练代码、模型实现、训练脚本
- `ml/configs`：实验配置和公共配置
- `ml/artifacts`：checkpoint、metrics、历史实验输出
- `ml/inference`：后续放推理封装和模型加载接口
- `data/datasets`：原始数据和处理后数据
- `data/cache`：文本向量、图像向量等缓存
- `backend`：API 服务
- `frontend`：Web 前端
- `docs`：项目文档

## 当前训练入口

安装依赖：

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

构建 OpenBG-IMG 文本缓存：

```bash
python ml/training/scripts/build_cache_openbg_img_text.py ^
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir data/cache/openbg_img
```

构建 OpenBG-IMG 图像缓存：

```bash
python ml/training/scripts/build_cache_openbg_img_image.py ^
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root data/datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir data/cache/openbg_img
```

启动训练：

```bash
python ml/training/scripts/run_train.py ^
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml ^
  --common ml/configs/common.yaml
```

训练产物默认输出到：

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/
```
