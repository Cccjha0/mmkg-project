# MMKG Project Handover

This document describes how to install, prepare, and run the current MMKG demo project.

## 1. Repository

```powershell
git clone https://github.com/Cccjha0/mmkg-project.git
cd mmkg-project
```

The current application has three runtime pieces:

- FastAPI backend on `http://127.0.0.1:8000`
- Flask knowledge-graph search service on `http://127.0.0.1:5000`
- Vite React frontend on `http://localhost:3000`

## 2. Python Environment

Use Python from the project environment, then install root dependencies:

```powershell
cd E:\learn\R&D\mmkg-project
python -m pip install -r requirements.txt
```

Important Python dependencies include:

- FastAPI / Uvicorn for the main backend
- Flask / flask-cors for the 3D KG search service
- PyTorch and scikit-learn for inference and search
- pandas / tqdm for data conversion scripts

One-command setup is available.

Windows PowerShell:

```powershell
cd E:\learn\R&D\mmkg-project
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

macOS/Linux:

```bash
cd /path/to/mmkg-project
bash scripts/install.sh
```

These scripts install Python dependencies, install frontend dependencies, and generate KG processed files when raw data is available.

## 3. Frontend Environment

```powershell
cd E:\learn\R&D\mmkg-project\frontend
npm install
```

The frontend uses Vite + React. The 3D KG view depends on:

- `react-force-graph-3d`
- `three`
- `three-spritetext`

## 4. Required Data

Raw OpenBG-IMG files should be under:

```text
data/datasets/openbg_img/raw/
```

The key files are:

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

The KG Flask search service additionally needs:

```text
data/datasets/openbg_img/processed/data.csv
data/datasets/openbg_img/processed/metadata.json
```

Generate them with:

```powershell
cd E:\learn\R&D\mmkg-project\kg
python convert_openbg.py
python generate_metadata.py
```

The Flask service creates `search.pkl` automatically in the same processed directory on first use.

## 5. Production Models

Model checkpoints are expected locally under:

```text
ml/artifacts/production_models/
```

This directory is not suitable for normal Git commits because it contains large checkpoint files. Keep it local or distribute it through an artifact store.

For the default Attribute Completion model, the backend checks:

```text
ml/artifacts/production_models/gate+residual/
```

It automatically selects the best valid run using:

```text
ml/artifacts/plots/gate+residual/best_summary.csv
```

A valid run must contain:

```text
config_merged.json
best.ckpt
```

If `production_models/gate+residual/` has no valid run, the backend falls back to:

```text
ml/artifacts/outputs/openbg_img_gated_vec_res_rel/
```

Manual override is still supported:

```powershell
$env:MMKG_RUN_DIR="ml/artifacts/production_models/gate+residual/20260314_223034_seed2"
```

Other useful environment variables:

```powershell
$env:MMKG_MODEL_CODE="gate+residual"
$env:MMKG_MODEL_NAME="Residual+Gate"
$env:MMKG_DEVICE="cpu"
```

## 6. Start The Project

The simplest startup path is platform-specific.

Windows PowerShell:

```powershell
cd E:\learn\R&D\mmkg-project
powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```

This opens separate PowerShell windows for FastAPI, Flask KG search, and the frontend.

macOS/Linux:

```bash
cd /path/to/mmkg-project
bash scripts/start-dev.sh
```

This starts FastAPI, Flask KG search, and the frontend in the background. Logs and pid files are written under:

```text
.runtime_logs/
```

Stop them with:

```bash
bash scripts/stop-dev.sh
```

Manual startup is also supported. Open three terminals.

Terminal 1, FastAPI backend:

```powershell
cd E:\learn\R&D\mmkg-project\backend
uvicorn app.main:app --reload --port 8000
```

macOS/Linux equivalent:

```bash
cd /path/to/mmkg-project/backend
python3 -m uvicorn app.main:app --reload --port 8000
```

Terminal 2, Flask KG service:

```powershell
cd E:\learn\R&D\mmkg-project\backend
python flask_app.py
```

macOS/Linux equivalent:

```bash
cd /path/to/mmkg-project/backend
python3 flask_app.py
```

Terminal 3, frontend:

```powershell
cd E:\learn\R&D\mmkg-project\frontend
npm run dev
```

macOS/Linux equivalent:

```bash
cd /path/to/mmkg-project/frontend
npm run dev
```

Open:

```text
http://localhost:3000
```

## 7. What Each Service Does

FastAPI `8000` serves:

- model performance APIs
- entity detail APIs
- attribute completion APIs
- FastAPI observed subgraph APIs
- static OpenBG-IMG images

Flask `5000` serves the 3D Knowledge Graph page:

- `/search/<k>/<n>/<p>/<query>?lang=en`
- `/images/<entity_id>/<filename>`

The current frontend 3D KG page still calls Flask directly at `127.0.0.1:5000`.

## 8. Frontend Pages

### Model Performance

Uses FastAPI:

```text
/api/performance/overview
/api/performance/model-comparison
/api/performance/accuracy-curves
```

Accuracy curves read `ml/artifacts/plot_input/<model>/seed*.csv` when old `metrics.csv` outputs are unavailable.

### Attribute Completion

Uses FastAPI:

```text
/api/entities/<entity_id>
/api/entities/<entity_id>/attribute-completion?topk=5
```

Demo product IDs live in:

```text
frontend/src/data/demoProducts.ts
```

The list includes both image-backed and no-image entities.

### Knowledge Graph Explorer

Uses Flask:

```text
http://127.0.0.1:5000/search/...
```

The initial query is preset to:

```text
Pants
```

The backend limits 2-hop graph expansion to avoid browser freezes.

## 9. Validation Commands

Backend import check:

```powershell
cd E:\learn\R&D\mmkg-project\backend
python -c "import app.main; print('ok')"
```

Backend compile check:

```powershell
cd E:\learn\R&D\mmkg-project
python -m compileall backend/app
```

Frontend build check:

```powershell
cd E:\learn\R&D\mmkg-project\frontend
npm run build
```

The frontend build currently emits a large chunk warning because Three.js is bundled. This warning does not block running the app.

## 10. Troubleshooting

### `/api/entities/...` returns 503

The model is not ready. Check:

```powershell
cd E:\learn\R&D\mmkg-project\backend
python -c "from app.services.inference_service import predictor_status; print(predictor_status())"
```

Make sure the selected run contains `config_merged.json` and `best.ckpt`.

### KG page is blank

Start the Flask service:

```powershell
cd E:\learn\R&D\mmkg-project\backend
python flask_app.py
```

Also check that these files exist:

```text
data/datasets/openbg_img/processed/data.csv
data/datasets/openbg_img/processed/metadata.json
```

### `Neighbours=2` freezes the browser

This should now be mitigated by backend graph limits and frontend input bounds. Restart Flask after pulling new code.

### Performance page returns 400 for accuracy curves

Check that `ml/artifacts/plot_input/` contains seed CSV files or that old metrics files exist under `ml/artifacts/outputs/`.

## 11. Git Notes

Do not commit large model checkpoints. `.gitignore` already ignores `*.ckpt`.

The local directory below may remain untracked:

```text
ml/artifacts/production_models/
```

That is expected unless the project adopts Git LFS or another artifact distribution mechanism.
