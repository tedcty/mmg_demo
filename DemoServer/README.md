# DemoServer — MMG Outreach Hub

Flask app that serves the outreach demo landing page (`demo.html`) at `/` and
hosts the **SSM Shoulder Predictor** demo at `/ssm/`, backed by a Python
prediction pipeline. It also hosts the **Hear Your Muscles** EMG game at `/emg/`
— a browser (Web Audio + Canvas) port of `Demos/Spikerbox-EMG`, served straight
from that folder with no build step. HTTPS is on by default: the app runs on
**https://0.0.0.0:8443**, and **http://…:8000 auto-redirects** to it (so typing
`host:8000` still works) — see [Tablets & offline use](#tablets--offline-use).

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
python server.py                # HTTPS :8443 (+ http :8000 redirect); --http for plain HTTP
python server.py --check        # preflight doctor only: checks env/frontend/cert/ports
```

## Requirements

- **Python 3.12** (conda recommended)
- **Node.js + npm** (for the frontend build) — auto-installed into the conda
  env by `setup_demo_server.py` if not already on your PATH
- Python deps pinned in [`requirements.txt`](requirements.txt): Flask +
  flask-cors for the server, `zeroconf` for the `mmg-demo.local` mDNS name, and
  the SSM stack (numpy, pandas, scipy, scikit-learn, vtk, plus `gias3` and
  `ptb_mmg`). All are on PyPI.

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

## Troubleshooting — `doctor.py`

When something's wrong, run the doctor:

```bash
python DemoServer/doctor.py            # diagnose (read-only)
python DemoServer/doctor.py --reset    # stop any running server / free ports 8443 + 8000
python DemoServer/doctor.py --restart  # reset, then start a fresh server
python DemoServer/doctor.py --http     # diagnose/reset for plain-HTTP mode (:8000)
```

It checks prerequisites (dist, model/data, bones.json, TLS cert), reports which
PID (if any) is holding the ports, probes that each port speaks the right
protocol, and prints the correct URLs for this machine and for tablets. It's
cross-platform and needs no arguments to get a full report.

Two problems it exists to solve:

- **"Port already in use" / preflight FAIL** — a previous server is still
  running. `doctor.py --reset` stops it and frees the ports.
- **`ERR_CONNECTION_RESET` in the browser** — you're on the wrong
  protocol/port. The rule is strict:

  | URL | Result |
  |-----|--------|
  | `https://<host>:8443` | ✅ the app |
  | `http://<host>:8000`  | ✅ redirects to 8443 |
  | `http://<host>:8443`  | ❌ reset (plain HTTP on the TLS port) |
  | `https://<host>:8000` | ❌ reset (TLS on the plain-HTTP port) |

### GUI dashboard

Prefer a window over the command line? Run the Material-styled control panel:

```bash
conda run -n demo python DemoServer/dashboard.py
```

It shows live status (running state, PID, uptime, port/prereq/protocol
indicators, and CPU/GPU load with a **server-process** breakdown — the whole
server tree including SSM prediction subprocesses — under the machine total) and
a prominent **Tablet access** banner with the
`https://mmg-demo.local:8443` address, a **scan-to-open QR code**, and a **Copy**
button, with **Start / Stop / Restart / Reset ports** buttons, a **Run doctor**
button (prints the full diagnostic report into the log pane), a server-log pane,
and a toggle for auto-refresh. The **Access** tab adds larger QR codes — one to
open the demo and one for the **`/trust`** page (with **Open /trust** / **Copy**
buttons) to onboard a new tablet's cert. A **Name** diagnostics row verifies that
`mmg-demo.local` actually resolves on the network (so you know before a visitor
does whether to fall back to the IP). Any QR can be **clicked to enlarge** into a
scannable pop-up. The **Demos** card has a **Rebuild frontend** button that runs
`vite build --base /ssm/` (streaming into the console) so you can fix a stale
`REBUILD` without a terminal, and if the server process exits unexpectedly a
dismissable **crash banner** appears. A **Keep alive** toggle auto-restarts the
server after an unexpected exit (with backoff, giving up after a few rapid
crashes) for unattended events. Launch flags support the same:
`dashboard.py --autostart` starts the server as soon as the panel opens, and
`--keep-alive` enables the watchdog — together they make an "open the laptop and
everything's up" kiosk. Buttons have tooltips and keyboard shortcuts
(**F5** refresh, **Ctrl+S** start, **Ctrl+K** stop, **Ctrl+R** restart,
**Ctrl+Shift+R** rebuild, **Ctrl+D** doctor, **Alt+1…4** switch pages). The
status icon and chip track state live —
green when running, amber while **STARTING / STOPPING** — with a progress bar
during those transitions, driven by the server actually answering (not a fixed
delay). The type scale, buttons and window are sized for **13–16" portable
outreach laptops** — it opens fit-to-screen and the KPI cards reflow onto extra
rows when the window is narrow, so nothing clips.

It's the same `doctor.py` logic behind a PySide6 + qt-material UI, so it reflects
any server — even one you started from a terminal or left running when you closed
a previous panel. It **adopts** an already-running server on open:
status, **Stop / Restart / Reset**, and the **crash banner + Keep alive**
watchdog all work on it (control is by port/PID, so it doesn't need to be the
process this panel spawned). The one thing it can't recover is the **live console
log** for a server it didn't launch — status and control still work, there's just
no captured output. Run it in the `demo` env so it launches the server with the
right interpreter.

## Tablets & offline use

For outreach events on iPad/Android tablets, note:

- **Offline-ready hub.** Vue and Tailwind are **vendored** in
  `resources/vendor/` and served locally, so the landing page (`/`) and the
  demos work on a LAN with **no internet**. Nothing loads from a public CDN.
- **`/ssm/`** (three.js/WebGL) and the info PDFs work on iOS and Android.
- **`/emg/` needs HTTPS for the microphone.** Browsers block `getUserMedia`
  on a non-`localhost` HTTP origin. The EMG audio worklet is served as a file
  (`web/emg-worklet.js`, not a `blob:` URL) so it also works on iOS/iPad.

### Reaching the server (`mmg-demo.local`)

The server advertises itself over **mDNS / Bonjour** as a fixed name, so tablets
can use **`https://mmg-demo.local:8443`** instead of a LAN IP that changes with
DHCP. This needs no router config and works fully offline — the `zeroconf`
package (in `requirements.txt`) bundles its own responder, so the host PC needs
nothing extra installed.

- Resolves natively on **iOS/iPadOS**, **macOS**, **Windows 10+**, and **Linux
  with Avahi** (installed by default on most desktops).
- **Android is unreliable** with `.local`, so the server also prints its **LAN
  IP** on startup as a fallback — the dashboard's *Tablet access* banner shows
  the name with the IP as a fallback hint.
- The name is baked into the cert SANs (below), so it's warning-free once the
  cert is trusted.

To use a different name, change `MDNS_HOST` in `server.py` (keep the matching
`MDNS_FQDN` in `doctor.py` and `setup_https.py` in sync).

### HTTPS (on by default)

The server serves **HTTPS on port 8443 by default**, plus a plain-HTTP
redirector on **:8000** that 302s to HTTPS — so `http://<host>:8000` (what
browsers assume when you type `host:8000`) auto-upgrades instead of resetting.
On first run it auto-generates a persistent self-signed cert in `certs/`
(git-ignored) whose SANs include `localhost`, `mmg-demo.local`, `127.0.0.1` and
the detected LAN IP. Requires the `cryptography` package (in `requirements.txt`).
Use `--http` to opt out (plain HTTP on :8000, mic then only on localhost);
`--https-port` / `--http-port` change the ports.

Open the tablet at **`https://mmg-demo.local:8443`** (see
[Reaching the server](#reaching-the-server-mmg-demolocal)), or use the LAN IP:
**`http://<server-ip>:8000`** (redirects) or **`https://<server-ip>:8443`**
directly. A self-signed cert triggers a browser warning:

- **Android Chrome** — tap **Advanced → Proceed**; the mic then prompts normally.
- **iOS/iPad Safari** — Safari will *not* grant the mic to an untrusted cert
  even after "visit anyway", so the cert must be **trusted once per iPad**.

**Warning-free — one command:**

```bash
python DemoServer/setup_https.py        # installs a local CA + writes certs/
```

This runs `mkcert` (auto-installed via conda-forge if needed): it trusts a local
CA on **this** machine and writes `certs/cert.pem` + `certs/key.pem` for
`localhost`, `mmg-demo.local`, `127.0.0.1` and your LAN IP. `server.py` prefers
those over the self-signed cert, so **this PC and Android show no warning** after
a restart. Pass extra hostnames/IPs as arguments if needed. Equivalent manual
steps:

```bash
mkcert -install
mkcert -cert-file DemoServer/certs/cert.pem -key-file DemoServer/certs/key.pem \
       mmg-demo.local <server-lan-ip> localhost 127.0.0.1
```

### Trusting tablets (self-serve)

Every tablet that runs the **EMG game** (SpikerBox in its mic input) needs the
CA installed once. `setup_https.py` copies the root CA next to the certs, and
the server hands it out with instructions at a plain-HTTP page so tablets can
reach it **before** they trust the cert:

- On the tablet, open **`http://<server-ip>:8000/trust`** → tap **Download the
  certificate** → follow the per-OS steps shown (auto-selected for iOS / Android
  / Windows). iOS needs the extra **Settings → General → About → Certificate
  Trust Settings** toggle.

**Do this once per tablet when you prep the kit**, not at every event: because
you own the tablets and the mkcert CA lasts ~10 years, a tablet trusted once
stays trusted. The **`mmg-demo.local`** name already gives tablets a stable
address even if the DHCP IP changes, and it's in the cert's SANs — so prefer it
on the tablets and you won't need to re-issue the cert when the IP moves. (A
static LAN IP or DHCP reservation for the server is still a good backstop for
Android, which may not resolve `.local`.) If you do re-issue the cert for a new
IP, re-run `python DemoServer/setup_https.py <new-ip>` — `mmg-demo.local` is
always included automatically.

> **Windows tablet clients** are the exception: installing a root CA there is a
> manual Certificate Import Wizard step and may be **blocked by policy** on
> managed devices. Prefer iOS/Android for EMG tablets, or pre-install the CA.
>
> A single self-contained machine (server + browser) needs none of this —
> `http://localhost:8000` is always a secure context.
