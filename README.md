# OpenBG Link Prediction Framework

Configurable training framework for OpenBG-IMG link prediction experiments.

Current mainline model is `openbg_img_gated` with:
- relation-aware gated fusion
- optional entity residual branch
- self-adversarial negative sampling
- tail-only filtered evaluation

## Repository Layout

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

## 1) Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Dataset Placement

Put OpenBG-IMG source files under:

```text
datasets/openbg_img/raw/
```

Required:
- `OpenBG-IMG_train.tsv`
- `OpenBG-IMG_dev.tsv`
- `OpenBG-IMG_test.tsv`
- `OpenBG-IMG_entity2text.tsv`
- `OpenBG-IMG_relation2text.tsv`
- `OpenBG-IMG_images/` (e.g. `ent_xxxxxx/image_0.jpg`)

## 3) Build Feature Cache

Text cache:

```bash
python scripts/build_cache_openbg_img_text.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir cache/openbg_img
```

Image cache:

```bash
python scripts/build_cache_openbg_img_image.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir cache/openbg_img
```

Expected cache files:
- `cache/openbg_img/text_emb.pt`
- `cache/openbg_img/has_text.pt`
- `cache/openbg_img/img_emb.pt`
- `cache/openbg_img/has_img.pt`

## 4) Train

Main config (gate + residual):

```bash
python scripts/run_train.py --config configs/openbg_img_gated_vec_res_rel.yaml --common configs/common.yaml
```

Ablation configs:
- gate-only: `configs/openbg_img_gate_only.yaml`
- residual-only: `configs/openbg_img_residual_only.yaml`

## 5) Run Multi-Seed

Change `system.seed` in `configs/common.yaml`, then run repeatedly, or generate temp common files.

Example (PowerShell, one line):

```powershell
$seeds=1,2,3; foreach($s in $seeds){ $tmp="configs/common_seed$s.yaml"; ((Get-Content configs/common.yaml -Raw) -replace 'seed:\s*\d+', "seed: $s") | Set-Content $tmp -Encoding utf8; Write-Host "=== Running seed $s ==="; python scripts/run_train.py --config configs/openbg_img_gated_vec_res_rel.yaml --common $tmp }
```

## 6) Outputs

Run directory format:

```text
outputs/<exp_name>/<timestamp>_seed<seed>/
```

Typical files:
- `best.ckpt`
- `config_merged.json`
- metrics csv (name can vary by trainer setting)

Metrics usually include:
- `mrr`, `hits@1`, `hits@3`, `hits@10`
- gated runs may include gate stats:
  `g_mean_all`, `g_std_all`, `g_mean_img`, `g_std_img`, `g_mean_noimg`, `g_std_noimg`, `g_frac_img_in_sample`

## 7) Plotting Scripts

`scripts/plot_kg_results.py` expects standard training metrics and gate stats columns for gate plots.

If your CSV does not contain gate-related columns, gate plotting parts will fail unless you guard or remove those calls.

`scripts/build_model_comparison_bar_chart.py` is a manual plotting helper (you should edit the hardcoded values before use).

## 8) Extension Guide for Teammates

If a teammate adds a new model:
1. Implement model in `src/models/...` with `forward(pos, neg)` and `score(triples)`.
2. Register it in `src/models/build_model.py` under a new `model.name`.
3. Add a new experiment config in `configs/`.
4. If extra preprocessing is needed, add scripts under `scripts/` and store outputs under `cache/...`.
