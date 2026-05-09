# 实验结果整理草稿

本文档用于整理当前论文/报告实验结果，覆盖主结果表、baseline 对比、消融实验、图表、简短分析和结果文件备份。当前数值优先来自 `ml/artifacts/outputs/results/best_summary.csv`；其余实验目录中的散点结果暂作为候选结果，正式写论文前需要统一确认配置、seed、dev/test split 和评价协议。

## Checklist

- [x] 主结果表
- [x] baseline 对比
- [x] 消融实验
- [x] 图表
- [x] 简短分析
- [x] 结果文件备份

## 1. 主结果表

当前建议优先使用的一组 baseline 汇总结果来自：

```text
ml/artifacts/outputs/metrics/
```

该目录按模型和 seed 存放指标文件，包括 `text_only`、`text_rgcn`、`early`、`gate_only` 和 `gate+residual`。下面结果均取每个 seed 中 dev MRR 最高的 epoch，再计算 3 个 seed 的均值和标准差。

| Model | Seeds | MRR | Hits@1 | Hits@3 | Hits@10 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Text-only | 3 | 0.2218 ± 0.0114 | 0.1483 ± 0.0080 | 0.2414 ± 0.0115 | 0.3684 ± 0.0230 | 0.0411 ± 0.0049 |
| Text-RGCN | 3 | 0.2802 ± 0.0149 | 0.1717 ± 0.0071 | 0.3177 ± 0.0203 | 0.5040 ± 0.0347 | 0.0078 ± 0.0007 |
| Early Fusion | 3 | 0.2960 ± 0.0028 | 0.1641 ± 0.0054 | 0.3475 ± 0.0011 | 0.5657 ± 0.0009 | 1.1242 ± 0.0152 |
| Gate-only | 3 | 0.3612 ± 0.0111 | 0.2181 ± 0.0169 | 0.4311 ± 0.0098 | 0.6483 ± 0.0058 | 1.0754 ± 0.0310 |
| Gate + Residual / Full Model | 3 | 0.4981 ± 0.0158 | 0.3505 ± 0.0142 | 0.5873 ± 0.0173 | 0.7819 ± 0.0210 | 0.7329 ± 0.1541 |

论文表格可写为：

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Text-only | 0.2218 ± 0.0114 | 0.1483 ± 0.0080 | 0.2414 ± 0.0115 | 0.3684 ± 0.0230 |
| Text-RGCN | 0.2802 ± 0.0149 | 0.1717 ± 0.0071 | 0.3177 ± 0.0203 | 0.5040 ± 0.0347 |
| Early Fusion | 0.2960 ± 0.0028 | 0.1641 ± 0.0054 | 0.3475 ± 0.0011 | 0.5657 ± 0.0009 |
| Gate-only | 0.3612 ± 0.0111 | 0.2181 ± 0.0169 | 0.4311 ± 0.0098 | 0.6483 ± 0.0058 |
| Gate + Residual / Full Model | 0.4981 ± 0.0158 | 0.3505 ± 0.0142 | 0.5873 ± 0.0173 | 0.7819 ± 0.0210 |

说明：

- 以上结果目前按 dev set 最佳 epoch 汇总。
- `Gate + Residual / Full Model` 对应 `ml/artifacts/outputs/metrics/gate+residual/`。
- 如果论文需要最终测试集结果，应在固定模型选择规则后，在 test set 上单独评估一次。
- 正式论文表格建议明确写出 `filtered tail prediction`，避免和双向 head/tail ranking 结果混淆。

## 2. Baseline 对比

### 2.1 推荐的正式 baseline 表

正式写作时建议至少放以下模型。`ml/artifacts/outputs/metrics/` 中已经包含 text-only、text-RGCN、early fusion、gate-only 和 gate+residual 的 3-seed 结果。

| Model | Config | MRR | Hits@1 | Hits@3 | Hits@10 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Text-only | metrics directory result | 0.2218 ± 0.0114 | 0.1483 ± 0.0080 | 0.2414 ± 0.0115 | 0.3684 ± 0.0230 | 已有 3-seed 汇总 |
| Text-RGCN | metrics directory result | 0.2802 ± 0.0149 | 0.1717 ± 0.0071 | 0.3177 ± 0.0203 | 0.5040 ± 0.0347 | 已有 3-seed 汇总 |
| Early Fusion | `ml/configs/openbg_img_early.yaml` | 0.2960 ± 0.0028 | 0.1641 ± 0.0054 | 0.3475 ± 0.0011 | 0.5657 ± 0.0009 | 已有 3-seed 汇总 |
| Gate-only | `ml/configs/openbg_img_gate_only.yaml` | 0.3612 ± 0.0111 | 0.2181 ± 0.0169 | 0.4311 ± 0.0098 | 0.6483 ± 0.0058 | 已有 3-seed 汇总 |
| Residual-only | `ml/configs/openbg_img_residual_only.yaml` | 0.5998* | 0.4856* | 0.6728* | 0.8042* | 单次候选结果，未在 metrics 汇总目录中 |
| Gate + Residual / Full Model | `ml/configs/openbg_img_gated_vec_res_rel.yaml` | 0.4981 ± 0.0158 | 0.3505 ± 0.0142 | 0.5873 ± 0.0173 | 0.7819 ± 0.0210 | 已有 3-seed 汇总 |

`*` 表示该结果来自旧输出目录中的单次最佳结果，而不是 `ml/artifacts/outputs/metrics/` 的统一 3-seed 汇总。正式论文中建议为 residual-only 补齐 3-seed 结果，或者明确声明其为单次消融结果。

### 2.2 当前已扫描到的候选 baseline/ablation 最佳结果

这些结果来自各 run 目录中的 `metrics*.csv`，只作为整理线索。由于不同 run 可能对应不同阶段的配置改动，正式表格前需要回看对应 `config_merged.json`。

| Experiment | Run | Best Epoch | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `text_only` | `metrics/text_only/seed1` | 45 | 0.2069 | 0.1410 | 0.2256 | 0.3360 |
| `text_only` | `metrics/text_only/seed2` | 130 | 0.2238 | 0.1444 | 0.2462 | 0.3828 |
| `text_only` | `metrics/text_only/seed3` | 100 | 0.2345 | 0.1594 | 0.2524 | 0.3864 |
| `text_rgcn` | `metrics/text_rgcn/seed1` | 130 | 0.2595 | 0.1620 | 0.2890 | 0.4550 |
| `text_rgcn` | `metrics/text_rgcn/seed2` | 110 | 0.2937 | 0.1790 | 0.3340 | 0.5300 |
| `text_rgcn` | `metrics/text_rgcn/seed3` | 85 | 0.2875 | 0.1740 | 0.3300 | 0.5270 |
| `early` | `metrics/early/seed1` | 35 | 0.2972 | 0.1658 | 0.3460 | 0.5662 |
| `early` | `metrics/early/seed2` | 80 | 0.2987 | 0.1696 | 0.3484 | 0.5644 |
| `early` | `metrics/early/seed3` | 40 | 0.2921 | 0.1568 | 0.3482 | 0.5664 |
| `gate_only` | `metrics/gate_only/seed1` | 70 | 0.3743 | 0.2354 | 0.4450 | 0.6508 |
| `gate_only` | `metrics/gate_only/seed2` | 170 | 0.3472 | 0.1952 | 0.4238 | 0.6538 |
| `gate_only` | `metrics/gate_only/seed3` | 50 | 0.3621 | 0.2238 | 0.4246 | 0.6402 |
| `gate+residual` | `metrics/gate+residual/seed1` | 50 | 0.4788 | 0.3348 | 0.5652 | 0.7546 |
| `gate+residual` | `metrics/gate+residual/seed2` | 16 | 0.5175 | 0.3692 | 0.6074 | 0.8056 |
| `gate+residual` | `metrics/gate+residual/seed3` | 30 | 0.4978 | 0.3474 | 0.5894 | 0.7854 |
| `openbg_img_residual_only` | `20260313_161055_seed1` | 29 | 0.5998 | 0.4856 | 0.6728 | 0.8042 |
| `openbg_img_residual_only` | `20260305_234647_seed1` | 5 | 0.5674 | 0.4556 | 0.6394 | 0.7688 |
| `openbg_img_residual_only` | `20260313_153733_seed1` | 5 | 0.5671 | 0.4554 | 0.6406 | 0.7676 |

关于 `Residual-only`：

- `residual_only` 的候选结果明显高于当前 3-seed full model 汇总，可能说明二者不是同一最终配置、同一实验批次或同一评价设置。
- 不建议直接把上表所有结果混放进论文主表。更稳妥的做法是重新按统一配置和统一 seed 运行 baseline/full model。
- 如果确认 `20260313_161055_seed1` 的配置就是最终消融配置，可以先在消融表里填单次结果；如果要写正式论文，建议至少补 3 个 seed 并报告 mean ± std。

## 3. 消融实验

建议消融问题围绕当前方法的核心结构部件展开：

1. Relation-aware gated fusion 是否有效。
2. Entity residual compensation 是否有效。
3. 二者联合使用是否优于单独使用。

### 3.1 结构消融表

| Variant | Fusion | Residual | Normalized Mix | MRR | Hits@1 | Hits@3 | Hits@10 | Interpretation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| Text-only | no | no | no | 0.2218 ± 0.0114 | 0.1483 ± 0.0080 | 0.2414 ± 0.0115 | 0.3684 ± 0.0230 | 测试仅文本语义表示的基础能力 |
| Early Fusion | static | no | no | 0.2960 ± 0.0028 | 0.1641 ± 0.0054 | 0.3475 ± 0.0011 | 0.5657 ± 0.0009 | 测试简单拼接/融合是否足够 |
| Gate-only | yes | no | no | 0.3612 ± 0.0111 | 0.2181 ± 0.0169 | 0.4311 ± 0.0098 | 0.6483 ± 0.0058 | 测试关系感知门控的单独贡献 |
| Residual-only | no | yes | no | 0.5998* | 0.4856* | 0.6728* | 0.8042* | 测试实体残差补偿的单独贡献 |
| Gate + Residual / Full Model | yes | yes | yes | 0.4981 ± 0.0158 | 0.3505 ± 0.0142 | 0.5873 ± 0.0173 | 0.7819 ± 0.0210 | 联合使用门控与残差补偿 |

`*` 表示当前只填入已扫描到的单次候选结果，来源为 `ml/artifacts/outputs/openbg_img_residual_only/20260313_161055_seed1/metrics.csv`。该结果需要和最终 full model 使用相同配置版本、相同 split、相同 seed 设置后再进入正式论文主表。

## 4. 图表

当前已有图表位于：

```text
ml/artifacts/outputs/results/
```

| Figure | File | Purpose |
| --- | --- | --- |
| MRR per seed | `ml/artifacts/outputs/results/mrr_per_seed.png` | 展示不同 seed 下最佳 MRR 的波动 |
| MRR mean/std | `ml/artifacts/outputs/results/mrr_mean_std.png` | 展示 MRR 均值和标准差 |
| Hits mean/std | `ml/artifacts/outputs/results/hits_mean_std.png` | 展示 Hits@K 的均值和标准差 |
| Loss mean/std | `ml/artifacts/outputs/results/loss_mean_std.png` | 展示训练 loss 的均值和方差 |
| Gate mean analysis | `ml/artifacts/outputs/results/gate_mean_analysis.png` | 分析门控均值变化 |
| Gate std analysis | `ml/artifacts/outputs/results/gate_std_analysis.png` | 分析门控离散程度 |
| Image fraction | `ml/artifacts/outputs/results/image_fraction.png` | 展示采样实体中有图实体比例 |

论文/报告建议使用：

1. 主结果表：使用 `best_summary.csv` 整理为 LaTeX/Markdown 表格。
2. 稳定性图：优先放 `mrr_per_seed.png` 或 `mrr_mean_std.png`。
3. 诊断图：放 `gate_mean_analysis.png` 和 `gate_std_analysis.png`，用于解释门控机制。
4. 训练过程图：如果篇幅允许，放 `loss_mean_std.png`。

Markdown 引用示例：

```markdown
![MRR per seed](../ml/artifacts/outputs/results/mrr_per_seed.png)
![Gate mean analysis](../ml/artifacts/outputs/results/gate_mean_analysis.png)
```

## 5. 简短分析

可以先写成以下版本，后续根据最终 baseline/test 结果再精修。

当前 `Gate + Residual / Full Model` 在 3 个随机种子上的平均 dev MRR 为 `0.4981 ± 0.0158`，Hits@10 为 `0.7819 ± 0.0210`，明显优于 Text-only、Text-RGCN、Early Fusion 和 Gate-only。结果说明关系感知门控与实体残差补偿联合使用时，模型能够更有效地利用多模态信息并提升链接预测性能。

从 baseline 对比看，Text-only 的 MRR 为 `0.2218 ± 0.0114`，Text-RGCN 提升到 `0.2802 ± 0.0149`，Early Fusion 进一步达到 `0.2960 ± 0.0028`。这说明文本结构建模和简单多模态融合都能带来收益，但提升幅度有限。Gate-only 的 MRR 达到 `0.3612 ± 0.0111`，相较 Early Fusion 有明显提升，表明关系感知的动态融合比固定融合更适合当前 MMKG 补全任务。

从机制分析角度看，门控相关图表可以用于说明模型并非简单固定融合文本和图像信息，而是在训练中学习不同实体/模态条件下的融合偏好。若后续能够补充有图实体与缺图实体的分组结果，可以进一步支撑“面向缺失视觉模态的鲁棒融合”这一叙事。

当前需要谨慎的是，已扫描到的 `residual_only` 单次候选结果高于 full model 的 3-seed 汇总。这可能来自不同配置批次或评价设置差异。正式论文中不应直接混用不同阶段的结果，而应统一配置、统一 seed、统一 dev/test 评价流程后再下结论。

## 6. 结果文件备份

### 6.1 需要备份的轻量结果

建议备份并可提交到 GitHub 的轻量文件：

- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/RESULTS_SUMMARY_DRAFT.md`
- `ml/artifacts/outputs/results/best_summary.csv`
- `ml/artifacts/outputs/results/*.png`
- 关键实验的 `config_merged.json`
- 关键实验的 `metrics*.csv`

由于当前 `.gitignore` 忽略了 `ml/artifacts/outputs/results/**`，如果需要提交汇总结果，需要显式调整 `.gitignore` 或把结果复制到一个可跟踪目录，例如：

```text
docs/results/
```

### 6.2 不建议提交到 GitHub 的大文件

不建议提交：

- `*.ckpt`
- `*.pt`
- `data/cache/**`
- 原始图片目录 `OpenBG-IMG_images/`
- 完整训练输出目录 `ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/`

这些文件可以放在本地硬盘、网盘、实验室服务器或 release artifact 中。

### 6.3 建议的备份结构

```text
docs/results/
  README.md
  best_summary.csv
  figures/
    mrr_per_seed.png
    mrr_mean_std.png
    hits_mean_std.png
    gate_mean_analysis.png
    gate_std_analysis.png
  configs/
    full_model_seed1_config_merged.json
    full_model_seed2_config_merged.json
    full_model_seed3_config_merged.json
  metrics/
    full_model_seed1_metrics.csv
    full_model_seed2_metrics.csv
    full_model_seed3_metrics.csv
```

### 6.4 最终提交前检查

- [ ] 主结果使用 test set 还是 dev set 已明确。
- [ ] 所有 baseline 使用相同 dataset split。
- [ ] 所有 baseline 使用相同 filtered ranking 协议。
- [ ] 多 seed 的 seed 列表明确。
- [ ] 每个结果能追溯到对应 config 和 metrics 文件。
- [ ] checkpoint 和 embedding cache 没有被加入 Git。
