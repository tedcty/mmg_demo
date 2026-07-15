# DemoServer — MMG Outreach Hub

Flask app that serves the outreach demo landing page (`demo.html`) at `/` and
hosts the **SSM Shoulder Predictor** demo at `/ssm/`, backed by a Python
prediction pipeline. It also hosts the **Hear Your Muscles** EMG game at `/emg/`
— a browser (Web Audio + Canvas) port of `Demos/Spikerbox-EMG`, served straight
from that folder with no build step. Runs on **https://0.0.0.0:8000** (HTTPS is
on by default — see [Tablets & offline use](#tablets--offline-use)).

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
   at `/ssm/`. If `npm` isn't on your PATH, Node.js is installed into the
   conda env automatically (via conda-forge — no sudo, works on macOS too).
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
python server.py                # → https://localhost:8000  (add --http for plain HTTP)
```

## Requirements

- **Python 3.12** (conda recommended)
- **Node.js + npm** (for the frontend build) — auto-installed into the conda
  env by `setup_demo_server.py` if not already on your PATH
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

## Tablets & offline use

For outreach events on iPad/Android tablets, note:

- **Offline-ready hub.** Vue and Tailwind are **vendored** in
  `resources/vendor/` and served locally, so the landing page (`/`) and the
  demos work on a LAN with **no internet**. Nothing loads from a public CDN.
- **`/ssm/`** (three.js/WebGL) and the info PDFs work on iOS and Android.
- **`/emg/` needs HTTPS for the microphone.** Browsers block `getUserMedia`
  on a non-`localhost` HTTP origin. The EMG audio worklet is served as a file
  (`web/emg-worklet.js`, not a `blob:` URL) so it also works on iOS/iPad.

### HTTPS (on by default)

The server serves **HTTPS on port 8000 by default**. On first run it
auto-generates a persistent self-signed cert in `certs/` (git-ignored) whose
SANs include `localhost`, `127.0.0.1` and the detected LAN IP. Requires the
`cryptography` package (in `requirements.txt`). Use `--http` to opt out.

Open the tablet at **`https://<server-ip>:8000`** (the launch banner prints the
exact URL). A self-signed cert triggers a browser warning:

- **Android Chrome** — tap **Advanced → Proceed**; the mic then prompts normally.
- **iOS/iPad Safari** — Safari will *not* grant the mic to an untrusted cert
  even after "visit anyway", so the cert must be **trusted once per iPad**.

**Warning-free on iPad — use `mkcert`:**

```bash
# once, on the server machine
mkcert -install
mkcert -cert-file DemoServer/certs/cert.pem -key-file DemoServer/certs/key.pem \
       <server-lan-ip> localhost 127.0.0.1
```

The server prefers `certs/cert.pem` + `certs/key.pem` if present. Then install
`mkcert`'s **root CA** (`mkcert -CAROOT` → `rootCA.pem`) on each iPad once:
AirDrop/email it → Settings → **General → VPN & Device Management** → install
the profile → **General → About → Certificate Trust Settings** → enable it.
After that, `https://<server-ip>:8000` is trusted and the mic just prompts.

> Simplest alternative: run the browser **on the same machine as the server**
> (`https://localhost:8000`) — localhost is always a secure context.
