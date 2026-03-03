# OpenBG 链接预测框架（中文说明）

本仓库是一个可配置的 OpenBG 链接预测实验框架。  
当前主线是 **OpenBG-IMG + 关系感知门控融合 + 实体残差分支**。

## 当前已实现内容

- 数据流程：OpenBG-IMG（`train/dev/test` 三元组 + 实体图片）
- 训练入口：`scripts/run_train.py`
- 模型构建入口：`src/models/build_model.py` 中的 `openbg_img_gated`
- 当前门控模型特性：
  - 向量门控（vector gate）
  - 关系偏置门控（relation-aware gate bias）
  - 实体残差分支（entity residual）
  - `softplus` 约束的正残差系数（residual scale）

## 仓库结构

```text
openbg-lp/
  configs/
    common.yaml
    openbg_img_gated.yaml
    openbg_img_gated_vec_res_final.yaml
    openbg_img_gated_vec_res_rel.yaml
  scripts/
    run_train.py
    build_cache_openbg_img_text.py
    build_cache_openbg_img_image.py
    debug/
  src/
    data/
    eval/
    models/
    train/
    utils/
  datasets/    # 通过 .gitkeep 保留目录结构
  cache/       # 通过 .gitkeep 保留目录结构
  outputs/     # 通过 .gitkeep 保留目录结构
```

## 1）环境准备

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2）数据集放置（必须放在 raw）

将 OpenBG-IMG 数据放到：

```text
datasets/openbg_img/raw/
```

至少需要：

- `OpenBG-IMG_train.tsv`
- `OpenBG-IMG_dev.tsv`
- `OpenBG-IMG_test.tsv`
- `OpenBG-IMG_entity2text.tsv`
- `OpenBG-IMG_relation2text.tsv`
- `OpenBG-IMG_images/`（格式：`ent_xxxxxx/image_0.jpg`）

说明：

- 原始数据统一放 `raw/`。
- 清洗后的中间文件放 `datasets/openbg_img/processed/`。

## 3）构建缓存特征（文本 + 图片）

先构建文本缓存：

```bash
python scripts/build_cache_openbg_img_text.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir cache/openbg_img
```

再构建图片缓存：

```bash
python scripts/build_cache_openbg_img_image.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir cache/openbg_img
```

完成后 `cache/openbg_img/` 里应有：

- `text_emb.pt`
- `has_text.pt`
- `img_emb.pt`
- `has_img.pt`

## 4）开始训练

推荐使用关系感知配置：

```bash
python scripts/run_train.py --config configs/openbg_img_gated_vec_res_rel.yaml --common configs/common.yaml
```

可选配置：

- `configs/openbg_img_gated.yaml`（基础版）
- `configs/openbg_img_gated_vec_res_final.yaml`（更长训练/更稳峰值搜索）

## 5）训练输出

每次运行输出到：

```text
outputs/<exp_name>/<timestamp>_seed<seed>/
```

常见文件：

- `metrics.csv`
- `best.ckpt`
- `config_merged.json`
- `common.yaml`（运行快照）
- `experiment.yaml`（运行快照）

`metrics.csv` 包含：

- 主指标：`mrr`, `hits@1`, `hits@3`, `hits@10`
- 门控统计：  
  `g_mean_all`, `g_std_all`, `g_mean_img`, `g_std_img`,  
  `g_mean_noimg`, `g_std_noimg`, `g_frac_img_in_sample`

## 6）组员如何扩展这个框架（重点）

例如你组里做 text-only 模型的同学，可以按下面步骤接入：

### A. 新增模型代码

- 把模型类放到 `src/models/...`
- 保持与训练器兼容的接口：
  - `forward(pos_triples, neg_triples) -> loss`
  - `score(triples) -> scores`

### B. 在 build_model 注册入口

- 修改 `src/models/build_model.py`
- 增加 `model.name` 分支，返回：
  - `model`
  - `num_entities`

如果不注册，`scripts/run_train.py` 无法根据配置创建模型。

### C. 新增配置文件

- 新建 `configs/<your_exp>.yaml`
- 至少包含：
  - 数据路径
  - `model.name`
  - 模型所需超参数
  - `output.exp_name`

运行时会自动合并 `common.yaml + 你的实验配置`。

### D. 如需额外预处理（如 text 编码）

- 在 `scripts/` 增加构建脚本
- 特征输出到 `cache/...`
- 在 `build_model.py` 对应分支里加载这些缓存

### E. Debug 脚本分离

- 一次性检查脚本放 `scripts/debug/`
- 不要把调试逻辑耦合进主训练入口

## 7）Git 跟踪与大文件说明

- 仓库通过 `.gitkeep` 仅保留目录骨架。
- 大体积数据/缓存/训练结果由 `.gitignore` 忽略。
- `outputs/openbg_img_gated` 保留目录，运行结果子目录默认不跟踪。
