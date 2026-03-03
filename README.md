# OpenBG Link Prediction (Gated Fusion)

This repository currently focuses on **OpenBG-IMG link prediction** with a
**text-image Gated Fusion** model.

## Current Scope

- Dataset: OpenBG-IMG
- Model: `openbg_img_gated`
- Training entry: `scripts/run_train.py`
- Config files:
  - `configs/openbg_img_gated.yaml`
  - `configs/common.yaml`

Other placeholder files (for teammate extensions) may exist but are not part of
the active training pipeline.

## Repository Structure

```text
openbg-lp/
  configs/
    common.yaml
    openbg_img_gated.yaml
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
  datasets/            # directory skeleton tracked with .gitkeep
  cache/               # directory skeleton tracked with .gitkeep
  outputs/             # directory skeleton tracked with .gitkeep
```

## Data Placement

Datasets are not included in this repository. Place files under:

```text
datasets/openbg_img/raw/
```

Required files:

- `OpenBG-IMG_train.tsv`
- `OpenBG-IMG_dev.tsv`
- `OpenBG-IMG_test.tsv`
- `OpenBG-IMG_entity2text.tsv`
- `OpenBG-IMG_relation2text.tsv`
- `OpenBG-IMG_images/` (`ent_xxxxxx/image_0.jpg`)

## Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1: Build Cache (Text + Image)

Build text embeddings cache:

```bash
python scripts/build_cache_openbg_img_text.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir cache/openbg_img
```

Build image embeddings cache:

```bash
python scripts/build_cache_openbg_img_image.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir cache/openbg_img
```

Expected cache artifacts in `cache/openbg_img/` include:

- `text_emb.pt`
- `has_text.pt`
- `img_emb.pt`
- `has_img.pt`

## Step 2: Train Gated Fusion Model

```bash
python scripts/run_train.py --config configs/openbg_img_gated.yaml --common configs/common.yaml
```

## Evaluation

Evaluation uses filtered ranking metrics:

- MRR
- Hits@1
- Hits@3
- Hits@10

## Outputs

Each run is written to:

```text
outputs/openbg_img_gated/<timestamp>_seed<seed>/
```

Typical files:

- `metrics.csv`
- `best.ckpt`
- `config_merged.json`
- `common.yaml` (copied snapshot)
- `experiment.yaml` (copied snapshot)

For gated models, `metrics.csv` also includes gate statistics columns
(`g_mean_all`, `g_std_all`, `g_mean_img`, `g_std_img`, `g_mean_noimg`,
`g_std_noimg`, `g_frac_img_in_sample`).

## Notes on Git Tracking

- Directory skeletons are tracked via `.gitkeep`.
- Large dataset/cache/output contents are ignored by `.gitignore`.
- `outputs/openbg_img_gated` keeps the directory itself, but run result folders
  remain untracked.
