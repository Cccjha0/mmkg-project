# OpenBG 链接预测框架（中文说明）

这是一个用于 OpenBG-IMG 链接预测实验的可配置训练框架。

当前主线模型是 `openbg_img_gated`，支持：
- 关系感知的门控融合（relation-aware gated fusion）
- 可选的实体残差分支（entity residual）
- Self-adversarial 负采样损失
- Tail-only filtered 评估

## 目录结构

```text
openbg-lp/
  configs/
    common.yaml
    common_seed1.yaml
    common_seed2.yaml
    openbg_img_gated_vec_res_rel.yaml
    openbg_img_gate_only.yaml
    openbg_img_residual_only.yaml
    ...
  scripts/
    run_train.py
    build_cache_openbg_img_text.py
    build_cache_openbg_img_image.py
    plot_kg_results.py
    build_model_comparison_bar_chart.py
  src/
    data/
    eval/
    models/
    train/
    utils/
  datasets/
  cache/
  outputs/
```

## 1) 环境准备

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 数据集放置

请将 OpenBG-IMG 原始文件放到：

```text
datasets/openbg_img/raw/
```

必须包含：
- `OpenBG-IMG_train.tsv`
- `OpenBG-IMG_dev.tsv`
- `OpenBG-IMG_test.tsv`
- `OpenBG-IMG_entity2text.tsv`
- `OpenBG-IMG_relation2text.tsv`
- `OpenBG-IMG_images/`（例如 `ent_xxxxxx/image_0.jpg`）

## 3) 构建特征缓存

构建文本缓存：

```bash
python scripts/build_cache_openbg_img_text.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir cache/openbg_img
```

构建图像缓存：

```bash
python scripts/build_cache_openbg_img_image.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir cache/openbg_img
```

期望生成：
- `cache/openbg_img/text_emb.pt`
- `cache/openbg_img/has_text.pt`
- `cache/openbg_img/img_emb.pt`
- `cache/openbg_img/has_img.pt`

## 4) 训练

主配置（gate + residual）：

```bash
python scripts/run_train.py --config configs/openbg_img_gated_vec_res_rel.yaml --common configs/common.yaml
```

消融配置：
- gate-only：`configs/openbg_img_gate_only.yaml`
- residual-only：`configs/openbg_img_residual_only.yaml`

## 5) 多 seed 顺序运行

你可以改 `configs/common.yaml` 里的 `system.seed` 多次运行，或临时生成 seed 配置。

PowerShell 一行命令示例：

```powershell
$seeds=1,2,3; foreach($s in $seeds){ $tmp="configs/common_seed$s.yaml"; ((Get-Content configs/common.yaml -Raw) -replace 'seed:\s*\d+', "seed: $s") | Set-Content $tmp -Encoding utf8; Write-Host "=== Running seed $s ==="; python scripts/run_train.py --config configs/openbg_img_gated_vec_res_rel.yaml --common $tmp }
```

## 6) 输出结果

每次运行输出目录格式：

```text
outputs/<exp_name>/<timestamp>_seed<seed>/
```

常见文件：
- `best.ckpt`
- `config_merged.json`
- 指标 CSV（文件名可能因 trainer 设置而不同）

常见指标列：
- `mrr`, `hits@1`, `hits@3`, `hits@10`
- gated 实验通常还会包含：
  `g_mean_all`, `g_std_all`, `g_mean_img`, `g_std_img`, `g_mean_noimg`, `g_std_noimg`, `g_frac_img_in_sample`

## 7) 绘图脚本说明

`scripts/plot_kg_results.py` 默认依赖标准训练指标，且 gate 相关图依赖 gate 统计列。

如果你的 CSV 不包含 gate 列，gate 图相关步骤会报错，需要在脚本里做缺列保护或跳过对应绘图。

`scripts/build_model_comparison_bar_chart.py` 是手动对比图脚本，使用前请先修改其中硬编码数值。

## 8) 给组员扩展模型的步骤

如果组员要接入新模型，建议按这 4 步：
1. 在 `src/models/...` 实现模型，接口保持 `forward(pos, neg)` 和 `score(triples)`。
2. 在 `src/models/build_model.py` 注册新的 `model.name` 分支。
3. 在 `configs/` 新建实验配置文件。
4. 如需额外预处理，在 `scripts/` 新建脚本并将产物放到 `cache/...`，再在 build 分支中读取。
