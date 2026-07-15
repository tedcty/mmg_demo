#!/usr/bin/env python3
"""
setup_demo_server.py — one-shot setup for the MMG Outreach Demo Server.

Prepares a fresh clone to run DemoServer/server.py end to end:

  1. Python  — create/update the conda env `demo` (Python 3.12) and install
               DemoServer/requirements.txt into it.
  2. Frontend — install Node deps and build the SSM Vite bundle into
               TauriGUI/dist (this is what server.py serves at /ssm/).
               If npm is not on PATH, Node.js is installed into the conda
               env automatically (via conda-forge, no sudo required).
  3. HTTPS   — generate a trusted cert via mkcert (delegates to setup_https.py)
               so the HTTPS server shows no browser warning. Non-fatal: if it
               fails, the server still serves HTTPS with a self-signed cert.
  4. Launch  — optionally start the server on https://0.0.0.0:8000.

Run from anywhere:
  python DemoServer/setup_demo_server.py            # setup only
  python DemoServer/setup_demo_server.py --run      # setup, then launch
  python DemoServer/setup_demo_server.py --run-only # skip setup, just launch
  python DemoServer/setup_demo_server.py --no-https # skip the mkcert cert step

The frontend dist is gitignored, so this build step is REQUIRED after every
fresh clone and after any change under TauriGUI/src.
"""

import argparse
import os
import shutil
import subprocess
import sys

ENV_NAME = "demo"
PY_VERSION = "3.12"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
GUI_DIR = os.path.join(REPO, "Demos", "SSM Demo", "predict_gui", "TauriGUI")
REQS = os.path.join(HERE, "requirements.txt")
SERVER = os.path.join(HERE, "server.py")
SETUP_HTTPS = os.path.join(HERE, "setup_https.py")


def find_conda():
    """Locate a runnable conda executable, even when it isn't on PATH.

    On Windows `conda` is often only a .bat in condabin/ (or a shell function),
    so `subprocess.run(["conda", ...])` fails with WinError 2 even though conda
    is installed. We look, in order, at:
      1. CONDA_EXE — the real exe path conda exports when a base/env is active;
      2. a plain PATH lookup (conda.exe first, then conda);
      3. the install derived from this interpreter (we're commonly launched by
         an env's own python, e.g. <base>/envs/<name>/python.exe).
    Prefer a .exe/binary over a .bat so it runs without a shell.
    """
    candidates = []
    env_exe = os.environ.get("CONDA_EXE")
    if env_exe:
        candidates.append(env_exe)
    candidates.append(shutil.which("conda.exe"))

    # Derive the base install from sys.prefix: if we're inside <base>/envs/<name>
    # the base is two levels up; otherwise sys.prefix may itself be the base.
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


CONDA = find_conda()


def run(cmd, cwd=None, check=True):
    """Run a command, streaming output. Returns True on success."""
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n$ {printable}")
    shell = not isinstance(cmd, list)
    # On Windows, npm/npx (and other .cmd/.bat shims) can't be launched by
    # CreateProcess directly — a bare "npm" 500s with WinError 2 even though it
    # is on PATH. If the first token doesn't resolve to a real .exe/.com, run it
    # through the shell so cmd.exe resolves the shim via PATHEXT.
    if isinstance(cmd, list) and os.name == "nt":
        resolved = shutil.which(cmd[0])
        if not resolved or not resolved.lower().endswith((".exe", ".com")):
            cmd = subprocess.list2cmdline(cmd)
            shell = True
    try:
        subprocess.run(cmd, cwd=cwd, shell=shell, check=check)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: command failed ({e.returncode})")
        return False
    except (FileNotFoundError, OSError) as e:
        exe = printable.split()[0] if printable else "command"
        print(f"  ERROR: could not run '{exe}': {e}")
        return False


def conda_env_exists(name):
    if not CONDA:
        return False
    try:
        out = subprocess.run(
            [CONDA, "env", "list"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    for line in out.splitlines():
        # Match the env name in the first column (ignore the active-env '*').
        parts = line.replace("*", " ").split()
        if parts and parts[0] == name:
            return True
    return False


def setup_python():
    print("\n" + "=" * 54)
    print(f"[1/3] Python environment  (conda env: {ENV_NAME})")
    print("=" * 54)

    if not CONDA:
        print("  WARNING: could not locate a conda executable.")
        print(f"  Install Miniconda, or create the env manually, then:")
        print(f"    pip install -r {REQS}")
        return False
    print(f"  Using conda: {CONDA}")

    if conda_env_exists(ENV_NAME):
        print(f"  Env '{ENV_NAME}' already exists — reusing it.")
    else:
        print(f"  Creating env '{ENV_NAME}' (Python {PY_VERSION})...")
        if not run([CONDA, "create", "-y", "-n", ENV_NAME, f"python={PY_VERSION}"]):
            return False

    print(f"  Installing DemoServer/requirements.txt into '{ENV_NAME}'...")
    ok = run([CONDA, "run", "-n", ENV_NAME, "pip", "install", "-r", REQS])
    if ok:
        print("  SUCCESS: Python dependencies installed.")
    else:
        print("  ERROR: dependency install failed. Check the pip output above —")
        print("  gias3 / ptb_mmg are specialised packages and may need retrying.")
    return ok


def ensure_node():
    """Make sure npm/npx are available, returning them as command-prefix lists.

    Prefers a system Node.js already on PATH. If none is found, installs
    Node.js into the conda env via conda-forge (no sudo, works the same on
    macOS/Linux/Windows) and returns `conda run`-prefixed commands.

    Returns (None, None) if Node cannot be provided.
    """
    if shutil.which("npm") and shutil.which("npx"):
        print("  Using Node.js already on PATH.")
        return ["npm"], ["npx"]

    print("  'npm' not found on PATH — attempting to install Node.js...")
    if CONDA and conda_env_exists(ENV_NAME):
        # conda-forge ships a recent Node (Vite needs 18+); the `defaults`
        # channel can lag, so pin the channel explicitly.
        if run([CONDA, "install", "-y", "-n", ENV_NAME, "-c", "conda-forge", "nodejs"]):
            prefix = [CONDA, "run", "--no-capture-output", "-n", ENV_NAME]
            if run(prefix + ["npm", "--version"], check=False):
                print(f"  SUCCESS: Node.js installed into the '{ENV_NAME}' env.")
                return prefix + ["npm"], prefix + ["npx"]
        print("  ERROR: automatic Node.js install failed.")
    else:
        print("  Cannot auto-install: the conda env is not available.")

    print("  Install Node.js manually from https://nodejs.org/ and re-run.")
    return None, None


def setup_frontend():
    print("\n" + "=" * 54)
    print("[2/3] SSM frontend build  (served by server.py at /ssm/)")
    print("=" * 54)

    npm_cmd, npx_cmd = ensure_node()
    if npm_cmd is None:
        return False

    lock = os.path.join(GUI_DIR, "package-lock.json")
    sub = ["ci"] if os.path.exists(lock) else ["install"]
    print(f"  Installing Node deps (npm {' '.join(sub)})...")
    if not run(npm_cmd + sub, cwd=GUI_DIR):
        return False

    print("  Building Vite bundle into dist/ ...")
    # --base /ssm/ MUST match the route server.py serves the bundle under.
    if not run(npx_cmd + ["vite", "build", "--base", "/ssm/"], cwd=GUI_DIR):
        return False

    print("  SUCCESS: frontend built to TauriGUI/dist.")
    return True


def setup_https():
    """Generate a trusted HTTPS cert via mkcert (delegates to setup_https.py).

    Non-fatal: if this fails or is skipped, the server still serves HTTPS with
    a self-signed cert (browsers just show a warning you can click through).
    Runs inside the `demo` env so mkcert can be auto-installed from conda-forge.
    """
    print("\n" + "=" * 54)
    print("[3/3] HTTPS certificate  (trusted via mkcert)")
    print("=" * 54)
    if not CONDA or not conda_env_exists(ENV_NAME):
        print("  Skipping — conda env unavailable. The server still serves HTTPS")
        print("  with a self-signed cert; run DemoServer/setup_https.py later to")
        print("  make it trusted.")
        return False
    ok = run([CONDA, "run", "--no-capture-output", "-n", ENV_NAME, "python", SETUP_HTTPS])
    if not ok:
        print("  HTTPS trust setup did not complete — the server still runs with")
        print("  a self-signed cert (accept the browser warning). You can re-run:")
        print("    python DemoServer/setup_https.py")
    return ok


def launch():
    print("\n" + "=" * 54)
    print("Launching DemoServer on port 8000  (HTTPS by default; Ctrl+C to stop)")
    print("=" * 54)
    if CONDA and conda_env_exists(ENV_NAME):
        cmd = [CONDA, "run", "--no-capture-output", "-n", ENV_NAME, "python", SERVER]
    else:
        print("  (conda env unavailable — launching with the current interpreter)")
        cmd = [sys.executable, SERVER]
    # cwd must be DemoServer so relative paths in server.py resolve.
    run(cmd, cwd=HERE, check=False)


def main():
    ap = argparse.ArgumentParser(description="Set up and/or launch the MMG Demo Server.")
    ap.add_argument("--run", action="store_true", help="launch the server after setup")
    ap.add_argument("--run-only", action="store_true", help="skip setup, just launch")
    ap.add_argument("--no-https", action="store_true",
                    help="skip the mkcert trusted-cert step (server still serves "
                         "HTTPS with a self-signed cert)")
    args = ap.parse_args()

    print("=" * 54)
    print("MMG Outreach Demo Server — Setup")
    print("=" * 54)

    if args.run_only:
        launch()
        return

    py_ok = setup_python()
    fe_ok = setup_frontend()
    https_ok = None if args.no_https else setup_https()

    print("\n" + "=" * 54)
    print("Setup summary")
    print("=" * 54)
    print(f"  Python env ({ENV_NAME}) : {'OK' if py_ok else 'INCOMPLETE — see above'}")
    print(f"  Frontend build          : {'OK' if fe_ok else 'INCOMPLETE — see above'}")
    if https_ok is None:
        print("  HTTPS trusted cert      : SKIPPED (self-signed cert in use)")
    else:
        print(f"  HTTPS trusted cert      : {'OK' if https_ok else 'INCOMPLETE — self-signed cert in use'}")
    print("\nStart the server with:")
    print(f"  conda run -n {ENV_NAME} python DemoServer/server.py")
    print("  # or: python DemoServer/setup_demo_server.py --run-only")

    if args.run and py_ok and fe_ok:
        launch()
    elif args.run:
        print("\nSkipping launch because setup did not complete cleanly.")


if __name__ == "__main__":
    main()
