from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_windows_install_script_references_expected_paths_and_commands() -> None:
    script = _read("scripts/install.ps1")

    assert "$FrontendDir" in script
    assert "$KgDir" in script
    assert "python -m pip install -r requirements.txt" in script
    assert "npm install" in script
    assert "python convert_openbg.py" in script
    assert "python generate_metadata.py" in script


def test_windows_start_script_declares_all_three_services() -> None:
    script = _read("scripts/start-dev.ps1")

    assert "uvicorn app.main:app --reload --port 8000" in script
    assert "python flask_app.py" in script
    assert "npm run dev" in script
    assert "MMKG FastAPI :8000" in script
    assert "MMKG KG Flask :5000" in script
    assert "MMKG Frontend :3000" in script


def test_unix_install_script_references_expected_paths_and_commands() -> None:
    script = _read("scripts/install.sh")

    assert "set -euo pipefail" in script
    assert "python3 -m pip install -r requirements.txt" in script
    assert "npm install" in script
    assert "python3 convert_openbg.py" in script
    assert "python3 generate_metadata.py" in script


def test_unix_start_and_stop_scripts_track_expected_services() -> None:
    start_script = _read("scripts/start-dev.sh")
    stop_script = _read("scripts/stop-dev.sh")

    assert "python3 -m uvicorn app.main:app --reload --port 8000" in start_script
    assert "python3 flask_app.py" in start_script
    assert "npm run dev" in start_script
    assert '"fastapi-8000"' in start_script
    assert '"kg-flask-5000"' in start_script
    assert '"frontend-3000"' in start_script
    assert '"$LOG_DIR/$name.pid"' in start_script
    assert 'for pidfile in "$LOG_DIR"/*.pid' in stop_script
    assert 'kill "$pid" || true' in stop_script
