# DemoServer — MMG Outreach Hub

Flask app that serves the outreach demo landing page (`demo.html`) at `/` and
hosts the **SSM Shoulder Predictor** demo at `/ssm/`, backed by a Python
prediction pipeline. Runs on **http://0.0.0.0:8000**.

> Not the same thing as the standalone Tauri desktop GUI. The Tauri app is set
> up by `Demos/SSM Demo/predict_gui/setup_project.py` and run via `run_app.py`.
> **This** server is set up by `setup_demo_server.py` in this folder.

## Quick start (fresh clone)

```bash
# from the repo root
python DemoServer/setup_demo_server.py --run
```

That single command:

1. Creates/updates the conda env **`demo`** (Python 3.12) and installs
   [`requirements.txt`](requirements.txt) into it.
2. Installs the frontend Node deps and **builds the Vite bundle** into
   `Demos/SSM Demo/predict_gui/TauriGUI/dist` — this is what the server serves
   at `/ssm/`.
3. Launches the server on port 8000.

Drop `--run` to set up without launching. Use `--run-only` to launch an
already-set-up checkout.

## Manual steps (equivalent)

```bash
# 1. Python env
conda create -n demo python=3.12
conda activate demo
pip install -r DemoServer/requirements.txt

# 2. Build the SSM frontend  (REQUIRED — dist/ is gitignored)
cd "Demos/SSM Demo/predict_gui/TauriGUI"
npm ci
npx vite build --base /ssm/     # --base MUST match the /ssm/ route

# 3. Run the server
cd ../../../../DemoServer
python server.py                # → http://localhost:8000
```

## Requirements

- **Python 3.12** (conda recommended)
- **Node.js + npm** (for the frontend build)
- Python deps pinned in [`requirements.txt`](requirements.txt): Flask +
  flask-cors for the server, and the SSM stack (numpy, pandas, scipy,
  scikit-learn, vtk, plus `gias3` and `ptb_mmg`). All are on PyPI.

## ⚠️ Rebuild the frontend after pulling

The served `/ssm/` app is a **pre-built** Vite bundle in `TauriGUI/dist`, and
`dist/` is **gitignored** — it is *not* regenerated automatically. After any
`git pull` (or edit) that touches `TauriGUI/src`, rebuild or the browser keeps
showing the old UI:

```bash
cd "Demos/SSM Demo/predict_gui/TauriGUI" && npx vite build --base /ssm/
```

Static assets are re-read per request, so a rebuild is picked up on the next
browser reload — no server restart needed. Restart the server only when
`server.py` or the Python pipeline changes.
