# Backend

The backend directory contains two runtime services:

- FastAPI app for model metadata, performance, entity, attribute-completion, graph, and prediction APIs.
- Flask service for the 3D knowledge-graph search page.

## FastAPI Service

Start from this directory:

```bash
python -m uvicorn app.main:app --reload --port 8000
```

OpenAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

Main router groups:

- `/api/health`
- `/api/meta/runtime`
- `/api/meta/model`
- `/api/performance/*`
- `/api/entities/{entity_id}`
- `/api/entities/{entity_id}/attribute-completion`
- `/api/graph/subgraph`
- `/api/predict/tail`

## Flask KG Search Service

Start from this directory:

```bash
python flask_app.py
```

The frontend KG visualization currently calls the Flask service directly at `http://127.0.0.1:5000`.

Main routes:

- `/search/<k>/<n>/<p>/<query>?lang=en`
- `/node_connections/<node_id>`
- `/images/<entity_id>/<filename>`

## Required Local Files

The FastAPI inference APIs use OpenBG-IMG dataset files and local model artifacts. Model checkpoints should be under:

```text
../ml/artifacts/production_models/
```

Attribute Completion selects a run automatically:

1. Check `../ml/artifacts/production_models/gate+residual/`.
2. Use `../ml/artifacts/plots/gate+residual/best_summary.csv` when available to pick the best valid seed/run.
3. Fall back to `../ml/artifacts/outputs/openbg_img_gated_vec_res_rel/`.
4. Respect `MMKG_RUN_DIR` when manually set.

A valid run needs:

```text
config_merged.json
best.ckpt
```

The Flask KG service needs:

```text
../data/datasets/openbg_img/processed/data.csv
../data/datasets/openbg_img/processed/metadata.json
```

Generate them with the scripts in `../kg/`.

## Useful Environment Variables

```bash
MMKG_RUN_DIR=../ml/artifacts/production_models/gate+residual/<run>
MMKG_MODEL_CODE=gate+residual
MMKG_MODEL_NAME=Residual+Gate
MMKG_DATASET=OpenBG-IMG
MMKG_DEVICE=cpu
```

On Windows PowerShell, use `$env:NAME="value"` instead.
