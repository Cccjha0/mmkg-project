# OpenBG Link Prediction Framework

This repository is a configurable framework for OpenBG link prediction
experiments.  
The current mainline is **OpenBG-IMG + relation-aware gated fusion + entity residual**.

## What Is Implemented Now

- Dataset pipeline: OpenBG-IMG (`train/dev/test` TSV + entity images)
- Training entry: `scripts/run_train.py`
- Active model builder path: `src/models/build_model.py` -> `openbg_img_gated`
- Current gated model features:
  - vector gate
  - relation-aware gate bias
  - entity residual branch
  - positive residual scale via `softplus`

## Repository Layout

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
  datasets/    # tracked as empty skeleton via .gitkeep
  cache/       # tracked as empty skeleton via .gitkeep
  outputs/     # tracked as empty skeleton via .gitkeep
```

## 1) Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Put Dataset Files in `raw`

Place OpenBG-IMG files under:

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

Notes:

- Keep original source files in `raw/`.
- If you need cleaned/intermediate artifacts, put them under
  `datasets/openbg_img/processed/`.

## 3) Build Feature Cache

Run text cache build:

```bash
python scripts/build_cache_openbg_img_text.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir cache/openbg_img
```

Run image cache build:

```bash
python scripts/build_cache_openbg_img_image.py ^
  --entity2text datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir cache/openbg_img
```

Expected artifacts in `cache/openbg_img/`:

- `text_emb.pt`
- `has_text.pt`
- `img_emb.pt`
- `has_img.pt`

## 4) Train

Recommended relation-aware run:

```bash
python scripts/run_train.py --config configs/openbg_img_gated_vec_res_rel.yaml --common configs/common.yaml
```

Alternative configs:

- `configs/openbg_img_gated.yaml` (baseline gated branch)
- `configs/openbg_img_gated_vec_res_final.yaml` (longer/more stable search setting)

## 5) Outputs

Each run is saved to:

```text
outputs/<exp_name>/<timestamp>_seed<seed>/
```

Typical files:

- `metrics.csv`
- `best.ckpt`
- `config_merged.json`
- `common.yaml` (snapshot)
- `experiment.yaml` (snapshot)

`metrics.csv` includes:

- `mrr`, `hits@1`, `hits@3`, `hits@10`
- gate statistics columns:
  `g_mean_all`, `g_std_all`, `g_mean_img`, `g_std_img`,
  `g_mean_noimg`, `g_std_noimg`, `g_frac_img_in_sample`

## 6) How Teammates Extend the Framework

This is the key workflow for teammates (for example, a text-model member).

### A. Add model code

- Put new model class in `src/models/...`
- Keep model API compatible with trainer:
  - `forward(pos_triples, neg_triples) -> loss`
  - `score(triples) -> scores`

### B. Register model in builder

- Edit `src/models/build_model.py`
- Add a new `model.name` branch that constructs your model and returns:
  - `model`
  - `num_entities`

Without this step, `run_train.py` cannot instantiate your model.

### C. Create experiment config

- Add `configs/<your_exp>.yaml`
- At minimum include:
  - dataset paths
  - `model.name`
  - model-specific fields
  - `output.exp_name`

`scripts/run_train.py` merges `common.yaml` + your exp config.

### D. If your model needs extra preprocessing

- Add a script under `scripts/` (or `scripts/debug/` for temporary checks)
- Save generated features to `cache/...`
- Reference those cache files in your model build branch

Example: text-only model members can add a dedicated text embedding cache script
and load it inside their `build_model` branch.

### E. Keep debug code separated

- Put one-off checks in `scripts/debug/`
- Avoid coupling debug scripts to the main training entry

## 7) Notes on Tracking and Large Files

- Directory skeletons are tracked via `.gitkeep`.
- Large dataset/cache/output contents are ignored by `.gitignore`.
- `outputs/openbg_img_gated` keeps directory structure, while run result folders
  remain untracked.
