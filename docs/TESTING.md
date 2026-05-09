# MMKG Testing Guide

This document describes the current test structure, what each layer covers, and how to run the tests during handover.

## 1. Test Layers

The project currently has six test layers:

| Layer | Command | Main Purpose |
| --- | --- | --- |
| Python API/unit tests | `python -m pytest` | Backend APIs, KG utilities, Flask KG routes, ML utility/model contracts, startup script checks |
| Frontend smoke test | `npm test` | Fast static checks for critical frontend defaults |
| Frontend component tests | `npm run test:components` | React component rendering and interaction checks |
| Frontend build | `npm run build` | Production bundle sanity check |
| Local E2E browser test | `npm run test:e2e` | Full demo flow across FastAPI, Flask KG, and Vite |
| Playwright browser install | `npm run test:e2e:install` | Installs Chromium for E2E tests |

The E2E test is a local acceptance test. It depends on real local data and a local browser install, so it is not the first test to run in a clean environment.

## 2. Install Test Dependencies

From the project root:

```bash
python -m pip install -r requirements-dev.txt
```

From the frontend directory:

```bash
cd frontend
npm install
```

For E2E tests, install Playwright Chromium once:

```bash
cd frontend
npm run test:e2e:install
```

If the browser download times out, rerun the same command later. The E2E test will fail with a clear missing-browser message until Chromium is installed.

## 3. Python Tests

Run from the project root:

```bash
python -m pytest
```

Current coverage:

- FastAPI smoke behavior and request validation
- Attribute Completion API metadata-only fallback
- Entity Detail API no-model behavior, image availability, 404 and 503 paths
- Performance metric parsing with mock `best_summary.csv` and `plot_input` files
- Production model run selection and fallback logic
- KG `data.csv` and `metadata.json` generation
- KG `Search` class shape and graph-size caps
- Flask KG `/search/...` and `/node_connections/...` routes
- Startup script static checks for Windows and macOS/Linux scripts
- ML filtered ranking behavior
- ML model shape checks for early fusion, gate+residual, gate-only, and residual-only variants

Python tests use small temporary fixtures and monkeypatching. They do not require full OpenBG-IMG data, real checkpoints, or model training.

## 4. Frontend Smoke Test

Run from `frontend/`:

```bash
npm test
```

This runs `frontend/scripts/smoke-test.mjs`.

It checks:

- `VITE_API_BASE_URL` fallback still points to `http://127.0.0.1:8000`
- Knowledge Graph default query remains `Pants`
- Knowledge Graph still calls the local Flask KG service at `http://127.0.0.1:5000`
- `.env.example` documents the FastAPI base URL

This test is intentionally lightweight and does not render React.

## 5. Frontend Component Tests

Run from `frontend/`:

```bash
npm run test:components
```

Current component coverage:

- `AttributeCompletion`
  - renders a no-image entity state
  - renders existing and predicted attribute rows
  - keeps predicted candidate switching interactive
- `ModelPerformance`
  - renders overview and comparison data from mocked hook state

The component tests use Vitest, jsdom, and Testing Library. The test setup includes a `ResizeObserver` mock for Recharts.

## 6. Frontend Build

Run from `frontend/`:

```bash
npm run build
```

The build currently passes. Vite may warn that the main bundle is larger than 500 kB. That warning does not fail the build, but it is a useful future optimization target.

## 7. Local E2E Test

Run from `frontend/` after installing Playwright Chromium:

```bash
npm run test:e2e:install
npm run test:e2e
```

Playwright starts three services through `frontend/playwright.config.ts`:

- FastAPI: `http://127.0.0.1:8000`
- Flask KG: `http://127.0.0.1:5000`
- Vite frontend: `http://127.0.0.1:3000`

The current E2E spec is:

```text
frontend/e2e/mmkg-demo.spec.ts
```

It verifies:

1. Model Performance page opens.
2. Attribute Completion page opens.
3. Knowledge Graph page opens.
4. The default Knowledge Graph query is `Pants`.
5. `Prune` can be set to `3`.
6. `Neighbours` can be set to `2`.
7. The `/search/1/2/3/Pants` response returns no more than 251 nodes and 500 links.
8. The Knowledge Graph page remains responsive after the query.

E2E requirements:

- Real OpenBG-IMG raw data under `data/datasets/openbg_img/raw/`
- Generated KG processed files:

```text
data/datasets/openbg_img/processed/data.csv
data/datasets/openbg_img/processed/metadata.json
```

- Playwright Chromium installed locally
- Ports `3000`, `5000`, and `8000` available, or compatible existing services already running

The E2E test is best treated as a local acceptance check before demo handover. It is heavier than the other tests and should not be required for every small code change.

## 8. Recommended Handover Test Order

For normal development:

```bash
python -m pytest
cd frontend
npm test
npm run test:components
npm run build
```

Before a demo or handover:

```bash
python -m pytest
cd frontend
npm test
npm run test:components
npm run build
npm run test:e2e
```

If E2E has never been run on the machine, run:

```bash
cd frontend
npm run test:e2e:install
```

## 9. Known Notes

- `python -m pytest` uses `.pytest_tmp/` as its temporary directory to avoid Windows user-temp permission issues.
- `.pytest_tmp/`, `frontend/test-results/`, and `frontend/playwright-report/` are ignored by Git.
- Large model artifacts and real datasets are not required for normal unit/API/component tests.
- The Attribute Completion tests intentionally cover metadata-only behavior because checkpoints may be absent on a fresh checkout.
- The Flask KG route tests use a fake `Search` object and do not require full `data.csv`.
- Playwright E2E is the only current test layer that expects real local data and browser installation.

## 10. Current Test Files

Python:

```text
tests/backend/test_api_smoke.py
tests/backend/test_attribute_entity_api.py
tests/backend/test_performance_service.py
tests/backend/test_runtime_selection.py
tests/kg/test_data_scripts.py
tests/kg/test_flask_routes.py
tests/kg/test_search.py
tests/ml/test_filtered_ranking.py
tests/ml/test_model_shapes.py
tests/scripts/test_startup_scripts.py
```

Frontend:

```text
frontend/scripts/smoke-test.mjs
frontend/src/test/AttributeCompletion.test.tsx
frontend/src/test/ModelPerformance.test.tsx
frontend/src/test/setup.ts
frontend/e2e/mmkg-demo.spec.ts
frontend/playwright.config.ts
```
