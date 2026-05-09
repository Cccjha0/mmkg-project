# Experiment Protocol and Reproduction Notes

This document records the current experimental setup for the MMKG project. It is intended to make the dataset split, training/evaluation settings, baselines, config files, and reproduction steps explicit.

## Checklist

- [x] dataset split
- [x] train/dev/test setting
- [x] negative sampling / evaluation protocol
- [x] metrics
- [x] baselines
- [x] config files
- [x] how to reproduce main experiments

## 1. Dataset Split

The repository currently contains two OpenBG-style datasets under `data/datasets/`.

### OpenBG-IMG

Main multimodal dataset used by the current OpenBG-IMG experiments.

| Split | File | Number of triples |
| --- | --- | ---: |
| Train | `data/datasets/openbg_img/raw/OpenBG-IMG_train.tsv` | 230,087 |
| Dev | `data/datasets/openbg_img/raw/OpenBG-IMG_dev.tsv` | 5,000 |
| Test | `data/datasets/openbg_img/raw/OpenBG-IMG_test.tsv` | 14,474 |

Auxiliary files:

- Entity text: `data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv`
- Relation text: `data/datasets/openbg_img/raw/OpenBG-IMG_relation2text.tsv`
- Optional English entity text: `data/datasets/openbg_img/raw/OpenBG-IMG_entity2text_en.tsv`
- Optional English relation text: `data/datasets/openbg_img/raw/OpenBG-IMG_relation2text_en.tsv`
- Entity images: `data/datasets/openbg_img/raw/OpenBG-IMG_images/`
- Cached features: `data/cache/openbg_img/`

The OpenBG-IMG model configs set `model.num_relations: 136`. The number of entities is inferred from cached text embeddings in `data/cache/openbg_img/text_emb.pt`.

### OpenBG500

Text-oriented dataset used by text-only and Text-RGCN experiments.

| Split | File | Number of triples |
| --- | --- | ---: |
| Train | `data/datasets/openbg500/raw/OpenBG500_train.tsv` | 1,242,550 |
| Dev | `data/datasets/openbg500/raw/OpenBG500_dev.tsv` | 5,000 |
| Test | `data/datasets/openbg500/raw/OpenBG500_test.tsv` | 5,000 |

Auxiliary files:

- Entity text: `data/datasets/openbg500/raw/OpenBG500_entity2text.tsv`
- Relation text: `data/datasets/openbg500/raw/OpenBG500_relation2text.tsv`
- Optional English entity text: `data/datasets/openbg500/raw/OpenBG500_entity2text_en.tsv`
- Optional English relation text: `data/datasets/openbg500/raw/OpenBG500_relation2text_en.tsv`
- Cached entity text embeddings: `data/cache/openbg500/entity_bert_emb.pt`

## 2. Train / Dev / Test Setting

Training is driven by `ml/training/scripts/run_train.py`.

The script currently:

1. Loads a shared config from `ml/configs/common.yaml`.
2. Loads an experiment config from `ml/configs/*.yaml`.
3. Merges the two configs.
4. Reads train and dev triples.
5. Builds filtered evaluation facts from `train + dev`.
6. Trains the model on the train split.
7. Evaluates periodically on the dev split.
8. Saves the best checkpoint by dev MRR.

Default shared training settings in `ml/configs/common.yaml`:

| Setting | Value |
| --- | --- |
| Embedding dimension | 256 |
| Learning rate | 0.001 |
| Batch size | 1024 |
| Negative ratio | 10 |
| Adversarial temperature | 2.0 |
| Epochs | 200 |
| Eval frequency | every 5 epochs |
| Early stopping patience | 10 eval rounds |
| Dev eval limit | 5,000 triples |
| Device | `cuda`, with fallback in code |
| Output root | `ml/artifacts/outputs` |

Experiment-specific configs may override these values. For example, the current full OpenBG-IMG model uses `lr: 0.0004`, `img_dropout: 0.2`, `chunk_size: 4096`, and `query_batch_size: 32`.

Test split note:

- The dataset configs include test file paths.
- The current training entrypoint evaluates dev during training and selects `best.ckpt` by dev MRR.
- For final reporting, test evaluation should be run once on the held-out test split after model selection. If using the current code as-is, add or use a small evaluation script that loads `best.ckpt`, builds filtering facts from `train + dev + test`, and reports test metrics on `test`.

## 3. Negative Sampling / Evaluation Protocol

### Negative Sampling During Training

Implemented in `ml/training/src/data/sampler.py`.

For each positive triple `(h, r, t)`:

- Repeat the positive triple `neg_ratio` times.
- With probability 0.5, corrupt the head entity.
- With probability 0.5, corrupt the tail entity.
- Replace the selected entity with a uniformly sampled random entity id from `[0, num_entities)`.

Current behavior:

- Training negatives are unfiltered.
- A sampled negative may accidentally be a known true triple.
- Default `neg_ratio` is 10 in `common.yaml`.

### Filtered Ranking Evaluation

Implemented in `ml/training/src/eval/filtered_ranking.py`.

Current evaluation computes filtered tail prediction only:

```text
(h, r, ?) ranking
```

For each evaluation triple `(h, r, t)`:

1. Score the gold triple `(h, r, t)`.
2. Score all candidate tail entities `(h, r, e)` for every entity `e`.
3. Filter other known true tails for the same `(h, r)` pair by assigning them `-inf`.
4. Keep the target tail unfiltered.
5. Compute the rank as:

```text
rank = 1 + number of unfiltered candidates with score > target_score
```

The current trainer builds filtering facts from `train + dev` for dev evaluation. For final test evaluation, use `train + dev + test` as the filtering set so that other known true triples in the test split are filtered correctly.

The evaluation code accepts `true_heads`, but current ranking logic only uses `true_tails`; head prediction is not included in the current reported metrics.

## 4. Metrics

Current reported metrics:

| Metric | Meaning |
| --- | --- |
| MRR | Mean Reciprocal Rank |
| Hits@1 | Fraction of queries whose target tail ranks in top 1 |
| Hits@3 | Fraction of queries whose target tail ranks in top 3 |
| Hits@10 | Fraction of queries whose target tail ranks in top 10 |

During training, metrics are written to:

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/metrics_seed1.csv
```

The run directory also stores:

- `best.ckpt`
- `config_merged.json`
- copied `common.yaml`
- copied experiment YAML

Additional diagnostic columns may be logged for gated models:

- `g_mean_all`
- `g_std_all`
- `g_mean_img`
- `g_std_img`
- `g_mean_noimg`
- `g_std_noimg`
- `g_frac_img_in_sample`

## 5. Baselines

The current repository supports the following model families/configurations.

| Role | Config | Model name | Notes |
| --- | --- | --- | --- |
| Text-only baseline | `ml/configs/text_exp_seed1.yaml` | `text_complex` | Text-initialized ComplEx on OpenBG500 |
| Text graph baseline | `ml/configs/text_rgcn.yaml` | `text_rgcn` | R-GCN with text embedding initialization on OpenBG500 |
| Early fusion baseline | `ml/configs/openbg_img_early.yaml` | `openbg_img_early` | OpenBG-IMG text/image early fusion |
| Gate-only ablation | `ml/configs/openbg_img_gate_only.yaml` | `openbg_img_gated` | Relation-aware gated fusion without residual |
| Residual-only ablation | `ml/configs/openbg_img_residual_only.yaml` | `openbg_img_gated` | Entity residual without fusion |
| Full model | `ml/configs/openbg_img_gated_vec_res_rel.yaml` | `openbg_img_gated` | Gated fusion + residual + normalized mix |

Recommended paper-facing comparison group for OpenBG-IMG:

1. Early Fusion
2. Gate-only
3. Residual-only
4. Full Model

Recommended additional baselines if time permits:

1. Text-only model on OpenBG-IMG using text embeddings only.
2. Image-only or image-dominant model, if meaningful for the dataset.
3. External MMKG completion baselines reproduced under the same split and filtered ranking protocol.

## 6. Config Files

Shared config:

- `ml/configs/common.yaml`

OpenBG-IMG configs:

- `ml/configs/openbg_img_early.yaml`
- `ml/configs/openbg_img_gate_only.yaml`
- `ml/configs/openbg_img_residual_only.yaml`
- `ml/configs/openbg_img_gated_vec_res_rel.yaml`

OpenBG500 / text configs:

- `ml/configs/text_exp_seed1.yaml`
- `ml/configs/text_rgcn.yaml`

Config loading:

- Implemented in `ml/training/src/utils/config.py`.
- The experiment config is loaded together with `common.yaml`.
- Experiment-specific keys override shared keys.

Important config fields:

| Section | Field | Purpose |
| --- | --- | --- |
| `dataset` | `train`, `dev`, `test` | Split file paths |
| `dataset` | `cache_dir` | Cached text/image features |
| `model` | `name` | Model builder dispatch key |
| `model` | `num_relations` | Relation count |
| `model` | `use_fusion` | Enable gated fusion branch |
| `model` | `use_residual` | Enable entity residual branch |
| `model` | `use_normalized_mix` | Normalize fusion/residual mixture weights |
| `training` | `neg_ratio` | Negatives per positive triple |
| `training` | `img_dropout` | Image modality dropout for OpenBG-IMG models |
| `training` | `fusion_warmup_epochs` | Residual-only warmup before fusion |
| `evaluation` | `chunk_size` | Number of candidate entities scored per chunk |
| `evaluation` | `query_batch_size` | Number of queries evaluated together |
| `system` | `seed` | Random seed |
| `system` | `device` | Requested device |
| `output` | `exp_name` | Output subdirectory name |

## 7. How to Reproduce Main Experiments

Run commands from the repository root.

### 7.1 Install Dependencies

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Build OpenBG-IMG Caches

Build text cache:

```powershell
python ml/training/scripts/build_cache_openbg_img_text.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --cache_dir data/cache/openbg_img
```

Build image cache:

```powershell
python ml/training/scripts/build_cache_openbg_img_image.py `
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv `
  --images_root data/datasets/openbg_img/raw/OpenBG-IMG_images `
  --cache_dir data/cache/openbg_img
```

Expected cache files include:

- `data/cache/openbg_img/text_emb.pt`
- `data/cache/openbg_img/img_emb.pt`
- `data/cache/openbg_img/has_img.pt`

### 7.3 Run OpenBG-IMG Baselines and Full Model

Early fusion:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_early.yaml `
  --common ml/configs/common.yaml
```

Gate-only:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gate_only.yaml `
  --common ml/configs/common.yaml
```

Residual-only:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_residual_only.yaml `
  --common ml/configs/common.yaml
```

Full model:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml `
  --common ml/configs/common.yaml
```

### 7.4 Run Text Baselines

Text ComplEx:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/text_exp_seed1.yaml `
  --common ml/configs/common.yaml
```

Text-RGCN:

```powershell
python ml/training/scripts/run_train.py `
  --config ml/configs/text_rgcn.yaml `
  --common ml/configs/common.yaml
```

### 7.5 Collect Results

Each run writes artifacts to:

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/
```

For each run, collect:

- Best dev MRR printed at the end of training.
- `metrics_seed1.csv` for epoch-level dev metrics.
- `best.ckpt` for final test evaluation.
- `config_merged.json` as the exact merged config snapshot.

For paper tables, report at minimum:

- MRR
- Hits@1
- Hits@3
- Hits@10

Recommended reporting format:

| Model | MRR | Hits@1 | Hits@3 | Hits@10 |
| --- | ---: | ---: | ---: | ---: |
| Early Fusion | TBD | TBD | TBD | TBD |
| Gate-only | TBD | TBD | TBD | TBD |
| Residual-only | TBD | TBD | TBD | TBD |
| Full Model | TBD | TBD | TBD | TBD |

### 7.6 Multi-seed Reproduction

For stable paper results, run at least three seeds, for example `1`, `2`, and `3`.

Current configs encode one seed at a time. To run multiple seeds:

1. Copy the target YAML config.
2. Change `system.seed`.
3. Optionally change `output.exp_name` to include the seed.
4. Run the same `run_train.py` command.

Report mean and standard deviation over seeds when enough runs are available.

## 8. Current Limitations to Track

- Training negatives are unfiltered.
- Current evaluation reports tail prediction only.
- Current `run_train.py` performs dev evaluation and checkpoint selection, but does not run final test evaluation automatically.
- `metrics_seed1.csv` is currently named with `seed1` even when another seed is used; the run directory still includes the actual seed in its name.
- For stronger paper claims, add multi-seed results, final test evaluation, and if possible external MMKG baselines under the same protocol.
