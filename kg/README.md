# KG Data Preparation

This directory contains scripts for generating the processed files used by the 3D knowledge-graph visualization.

## Inputs

Raw OpenBG-IMG files should exist under:

```text
../data/datasets/openbg_img/raw/
```

Important files:

```text
OpenBG-IMG_train.tsv
OpenBG-IMG_dev.tsv
OpenBG-IMG_test.tsv
OpenBG-IMG_entity2text.tsv
OpenBG-IMG_entity2text_en.tsv
OpenBG-IMG_relation2text.tsv
OpenBG-IMG_relation2text_en.tsv
OpenBG-IMG_images/
```

## Outputs

The KG Flask service reads:

```text
../data/datasets/openbg_img/processed/data.csv
../data/datasets/openbg_img/processed/metadata.json
```

`data.csv` contains the searchable graph edges. `metadata.json` contains display text, image availability, and lookup metadata used by the visualization.

## Generate Files

From this directory:

```bash
python convert_openbg.py
python generate_metadata.py
```

From the project root, the one-command setup scripts can also generate these files when raw data is present:

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

macOS/Linux:

```bash
bash scripts/install.sh
```

## Runtime Use

Start the Flask KG service from `backend/`:

```bash
python flask_app.py
```

The frontend KG page queries:

```text
http://127.0.0.1:5000/search/<k>/<n>/<p>/<query>?lang=en
```

The first search may also create a local `search.pkl` index in the processed data directory.
