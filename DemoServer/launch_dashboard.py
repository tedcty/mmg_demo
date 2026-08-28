#!/usr/bin/env python3
"""
launch_dashboard.py — double-click entry point for the MMG Demo control panel.

This is what "MMG Demo Launcher.exe" runs (build command in README.md). It
does a *quick* setup check — skipping anything already in place — then opens
the dashboard GUI:

  - conda env `demo`      — created (full setup) only if missing. Already
                             exists? Skipped entirely, no pip re-run.
  - SSM frontend build    — rebuilt only if missing or stale, per the same
                             check the dashboard's own "Rebuild frontend"
                             button uses (doctor.demo_status()).
  - HTTPS cert            — generated only if certs/cert.pem or key.pem is
                             missing.

This keeps a normal double-click fast (a couple of seconds once everything's
already set up) while still catching a fresh clone or a pulled frontend
change that needs (re)building. It does NOT redo a full
`pip install -r requirements.txt` on every launch — run
`python setup_demo_server.py` by hand after adding a new dependency.

Deliberately does not `import doctor` / `import setup_demo_server` at module
level: those modules locate repo files (server.py, TauriGUI/, certs/, ...)
relative to their own `__file__`, which breaks once bundled inside a frozen
exe. Instead this script stays a thin, freeze-safe bootstrapper and reaches
into them via `conda run ... python -c "..."` subprocesses, so they're
always read as real files on disk next to the exe, exactly as if you'd typed
the command yourself.
"""

import os
import shutil
import subprocess
import sys


def _find_repo_dir(start):
    """Locate the real DemoServer folder (the one with dashboard.py) starting
    from `start` and walking a few levels up.

    Covers the exe being run straight out of PyInstaller's own `dist/`
    output (one level below DemoServer/) instead of being moved/copied next
    to dashboard.py first — an easy step to skip, so don't require it."""
    d = start
    for _ in range(4):
        if os.path.exists(os.path.join(d, "dashboard.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return start


if getattr(sys, "frozen", False):
    HERE = _find_repo_dir(os.path.dirname(sys.executable))
else:
    HERE = _find_repo_dir(os.path.dirname(os.path.abspath(__file__)))

ENV_NAME = "demo"
SETUP_PY = os.path.join(HERE, "setup_demo_server.py")
DASHBOARD_PY = os.path.join(HERE, "dashboard.py")
CERT_FILE = os.path.join(HERE, "certs", "cert.pem")
KEY_FILE = os.path.join(HERE, "certs", "key.pem")


def _log(msg):
    print(f"[launcher] {msg}")


def find_conda():
    """Mirrors setup_demo_server.find_conda() (duplicated, not imported —
    see the module docstring for why)."""
    candidates = []
    env_exe = os.environ.get("CONDA_EXE")
    if env_exe:
        candidates.append(env_exe)
    candidates.append(shutil.which("conda.exe"))
    prefix = sys.prefix
    parent = os.path.dirname(prefix)
    bases = [prefix]
    if os.path.basename(parent).lower() == "envs":
        bases.append(os.path.dirname(parent))
    for base in bases:
        candidates.append(os.path.join(base, "Scripts", "conda.exe"))  # Windows
        candidates.append(os.path.join(base, "bin", "conda"))          # macOS/Linux
    candidates.append(shutil.which("conda"))
    for base in bases:
        candidates.append(os.path.join(base, "condabin", "conda.bat"))  # last resort
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def conda_env_exists(conda, name):
    try:
        out = subprocess.run([conda, "env", "list"], capture_output=True,
                              text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    for line in out.splitlines():
        parts = line.replace("*", " ").split()
        if parts and parts[0] == name:
            return True
    return False


def run_in(conda, env, args):
    cmd = [conda, "run", "--no-capture-output", "-n", env] + args
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=HERE)


def frontend_needs_build(conda):
    """Ask doctor.py's own staleness check (missing, or source changed since
    the last `vite build`) rather than re-deriving it here.

    Signals over stdout rather than exit code: `conda run` logs a scary-looking
    "ERROR ... failed" line to stderr for *any* nonzero exit from the wrapped
    process, which would fire on every ordinary "nothing to do" check. The
    probe must also stay a single line — this conda's `run` can't pass a
    `python -c` script containing newlines through to the subprocess."""
    probe = ("import doctor; "
             "d = next((x for x in doctor.demo_status() if x['route'] == '/ssm/'), None); "
             "print('NEEDS_BUILD' if d and d['status'] in ('build', 'stale') else 'UP_TO_DATE')")
    r = subprocess.run([conda, "run", "-n", "base", "python", "-c", probe],
                        cwd=HERE, capture_output=True, text=True)
    return "NEEDS_BUILD" in r.stdout


def check_mdns_readiness(conda):
    """Best-effort, read-only pre-flight check for whether mmg-demo.local is
    likely to work once the server starts — surfaced before the dashboard
    even opens, so you find out at launch time instead of when a visitor's
    tablet fails to connect. Purely informational: never blocks the launch,
    and the IP fallback works regardless of what this finds."""
    _log("Checking mDNS readiness...")
    probe = ("import doctor; "
             "print('ZC_OK' if doctor.zeroconf_available() else 'ZC_MISSING'); "
             "print('NET_UP' if not doctor.lan_ip().startswith('127.') else 'NET_DOWN'); "
             "h = doctor.mdns_blocked_hint(); "
             "print('HINT:' + h if h else 'HINT:')")
    r = subprocess.run([conda, "run", "-n", ENV_NAME, "python", "-c", probe],
                        cwd=HERE, capture_output=True, text=True)
    out = r.stdout

    if "ZC_MISSING" in out:
        _log("WARNING: the 'zeroconf' package isn't installed in the demo env — "
             "mmg-demo.local won't be advertised at all. Run setup_demo_server.py "
             "to install it; tablets can still use the IP in the meantime.")
        return
    if "NET_DOWN" in out:
        _log("WARNING: no network connection detected on this laptop — connect "
             "to WiFi (or turn on its Mobile Hotspot) before tablets can reach "
             "the demo at all, by name or by IP.")
        return

    hint_line = next((l for l in out.splitlines() if l.startswith("HINT:")), "HINT:")
    hint = hint_line[len("HINT:"):]
    if hint:
        _log(f"WARNING: mmg-demo.local likely won't resolve once the server starts — {hint}.")
    else:
        _log("mDNS looks ready (zeroconf installed, network connected, "
             "firewall not obviously blocking it).")


def rebuild_frontend(conda):
    probe = "import setup_demo_server as s, sys; sys.exit(0 if s.setup_frontend() else 1)"
    return run_in(conda, "base", ["python", "-c", probe]).returncode == 0


def ensure_https_cert(conda):
    probe = "import setup_demo_server as s; s.setup_https()"
    run_in(conda, "base", ["python", "-c", probe])  # non-fatal either way


def quick_setup(conda):
    if conda_env_exists(conda, ENV_NAME):
        _log(f"'{ENV_NAME}' env already exists — skipping Python setup.")
    else:
        _log(f"'{ENV_NAME}' env not found — running full setup (this can take a while)...")
        if run_in(conda, "base", ["python", SETUP_PY, "--no-https"]).returncode != 0:
            return False

    if frontend_needs_build(conda):
        _log("Frontend is missing or stale — rebuilding...")
        if not rebuild_frontend(conda):
            return False
    else:
        _log("Frontend build is up to date — skipping.")

    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        _log("HTTPS cert already present — skipping.")
    else:
        _log("No HTTPS cert found — generating one (non-fatal if this fails)...")
        ensure_https_cert(conda)

    check_mdns_readiness(conda)

    return True


def launch_dashboard(conda):
    _log("Opening the dashboard...")
    subprocess.run([conda, "run", "--no-capture-output", "-n", ENV_NAME,
                     "python", DASHBOARD_PY], cwd=HERE)


def main():
    print("=" * 54)
    print("MMG Demo Launcher")
    print("=" * 54)

    if not os.path.exists(DASHBOARD_PY):
        print(f"\nERROR: can't find dashboard.py near {HERE}.")
        print("Move/copy this exe into the DemoServer folder (next to dashboard.py,")
        print("setup_demo_server.py, doctor.py) and run it again.")
        input("\nPress Enter to close...")
        return

    conda = find_conda()
    if not conda:
        print("\nERROR: could not find a conda installation.")
        print("Install Miniconda (https://docs.conda.io/en/latest/miniconda.html),")
        print("then run this launcher again.")
        input("\nPress Enter to close...")
        return

    if quick_setup(conda):
        launch_dashboard(conda)
    else:
        print("\nSetup did not complete — see the errors above.")
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
