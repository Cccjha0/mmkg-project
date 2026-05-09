# MMKG Project

Multimodal commodity knowledge graph demo and research repository. The current project includes model training/inference code, a FastAPI backend, a Flask service for 3D knowledge-graph search, and a Vite React frontend.

For a fuller handover guide, see [docs/HANDOVER.md](docs/HANDOVER.md).

## Runtime Pieces

- FastAPI backend: `http://127.0.0.1:8000`
- Flask knowledge-graph search service: `http://127.0.0.1:5000`
- Vite React frontend: `http://localhost:3000`

## Repository Layout

```text
mmkg-project/
  backend/      FastAPI app and Flask KG search entrypoint
  frontend/     React + Vite web application
  kg/           OpenBG-IMG KG data conversion scripts
  ml/           training, inference, configs, and local artifacts
  data/         datasets, processed KG files, and embedding caches
  docs/         handover, experiment protocol, and project notes
  scripts/      one-command setup/start helpers
```

## Required Local Data

Raw OpenBG-IMG files should be placed under:

```text
data/datasets/openbg_img/raw/
```

The KG visualization also needs:

```text
data/datasets/openbg_img/processed/data.csv
data/datasets/openbg_img/processed/metadata.json
```

These processed files can be generated from `kg/`:

```bash
cd kg
python convert_openbg.py
python generate_metadata.py
```

Model checkpoints are expected locally under:

```text
ml/artifacts/production_models/
```

Large model artifacts are intentionally ignored by Git. The Attribute Completion backend first checks `ml/artifacts/production_models/gate+residual/` and then falls back to `ml/artifacts/outputs/openbg_img_gated_vec_res_rel/` when no production run is available.

## One-Command Setup

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

macOS/Linux:

```bash
bash scripts/install.sh
```

The setup scripts install Python dependencies, install frontend dependencies, and generate KG processed data when raw OpenBG-IMG files are available.

## Start The Demo

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-dev.ps1
```

macOS/Linux:

```bash
bash scripts/start-dev.sh
```

Then open:

```text
http://localhost:3000
```

On macOS/Linux, logs and pid files are written to `.runtime_logs/`. Stop background services with:

```bash
bash scripts/stop-dev.sh
```

## Manual Startup

Start three terminals if you prefer manual control.

FastAPI:

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

Flask KG service:

```bash
cd backend
python flask_app.py
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Useful Docs

- [docs/HANDOVER.md](docs/HANDOVER.md): complete install, startup, data, model, and troubleshooting guide
- [kg/README.md](kg/README.md): KG processed-data generation notes
- [backend/README.md](backend/README.md): backend services and endpoints
- [frontend/README.md](frontend/README.md): frontend development notes
- [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md): experiment and reproduction protocol
