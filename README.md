# MMKG Project

Multi-module repository for multimodal knowledge graph work. The current implementation focus is the ML training stack, with placeholders for inference, backend APIs, frontend web, shared data, and project docs.

## Repository Layout

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

## Current Status

- `ml/training`: training framework and model implementations
- `ml/configs`: experiment and shared config files
- `ml/artifacts`: training outputs and checkpoints
- `data/datasets`: raw and processed datasets
- `data/cache`: cached text and image embeddings
- `ml/inference`, `backend`, `frontend`, `docs`: scaffolded for the next phase

## Training Quick Start

Install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Build OpenBG-IMG text cache:

```bash
python ml/training/scripts/build_cache_openbg_img_text.py ^
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --cache_dir data/cache/openbg_img
```

Build OpenBG-IMG image cache:

```bash
python ml/training/scripts/build_cache_openbg_img_image.py ^
  --entity2text data/datasets/openbg_img/raw/OpenBG-IMG_entity2text.tsv ^
  --images_root data/datasets/openbg_img/raw/OpenBG-IMG_images ^
  --cache_dir data/cache/openbg_img
```

Run training:

```bash
python ml/training/scripts/run_train.py ^
  --config ml/configs/openbg_img_gated_vec_res_rel.yaml ^
  --common ml/configs/common.yaml
```

Artifacts are written under:

```text
ml/artifacts/outputs/<exp_name>/<timestamp>_seed<seed>/
```
