#!/usr/bin/env python3
"""
setup_demo_server.py — one-shot setup for the MMG Outreach Demo Server.

Prepares a fresh clone to run DemoServer/server.py end to end:

  1. Python  — create/update the conda env `demo` (Python 3.12) and install
               DemoServer/requirements.txt into it.
  2. Frontend — install Node deps and build the SSM Vite bundle into
               TauriGUI/dist (this is what server.py serves at /ssm/).
  3. Launch  — optionally start the server on http://0.0.0.0:8000.

Run from anywhere:
  python DemoServer/setup_demo_server.py            # setup only
  python DemoServer/setup_demo_server.py --run      # setup, then launch
  python DemoServer/setup_demo_server.py --run-only # skip setup, just launch

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


def run(cmd, cwd=None, check=True):
    """Run a command, streaming output. Returns True on success."""
    printable = " ".join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n$ {printable}")
    try:
        subprocess.run(cmd, cwd=cwd, shell=not isinstance(cmd, list), check=check)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: command failed ({e.returncode})")
        return False


def conda_env_exists(name):
    try:
        out = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    for line in out.splitlines():
        # Match the env name in the first column (ignore the active-env '*').
        parts = line.replace("*", " ").split()
        if parts and parts[0] == name:
            return True
    return False


def setup_python():
    print("\n" + "=" * 54)
    print(f"[1/2] Python environment  (conda env: {ENV_NAME})")
    print("=" * 54)

    if not shutil.which("conda"):
        print("  WARNING: 'conda' not found on PATH.")
        print(f"  Install Miniconda, or create the env manually, then:")
        print(f"    pip install -r {REQS}")
        return False

    if conda_env_exists(ENV_NAME):
        print(f"  Env '{ENV_NAME}' already exists — reusing it.")
    else:
        print(f"  Creating env '{ENV_NAME}' (Python {PY_VERSION})...")
        if not run(["conda", "create", "-y", "-n", ENV_NAME, f"python={PY_VERSION}"]):
            return False

    print(f"  Installing DemoServer/requirements.txt into '{ENV_NAME}'...")
    ok = run(["conda", "run", "-n", ENV_NAME, "pip", "install", "-r", REQS])
    if ok:
        print("  SUCCESS: Python dependencies installed.")
    else:
        print("  ERROR: dependency install failed. Check the pip output above —")
        print("  gias3 / ptb_mmg are specialised packages and may need retrying.")
    return ok


def setup_frontend():
    print("\n" + "=" * 54)
    print("[2/2] SSM frontend build  (served by server.py at /ssm/)")
    print("=" * 54)

    if not shutil.which("npm"):
        print("  ERROR: 'npm' not found. Install Node.js from https://nodejs.org/")
        return False

    lock = os.path.join(GUI_DIR, "package-lock.json")
    install_cmd = ["npm", "ci"] if os.path.exists(lock) else ["npm", "install"]
    print(f"  Installing Node deps ({' '.join(install_cmd)})...")
    if not run(install_cmd, cwd=GUI_DIR):
        return False

    print("  Building Vite bundle into dist/ ...")
    # --base /ssm/ MUST match the route server.py serves the bundle under.
    if not run(["npx", "vite", "build", "--base", "/ssm/"], cwd=GUI_DIR):
        return False

    print("  SUCCESS: frontend built to TauriGUI/dist.")
    return True


def launch():
    print("\n" + "=" * 54)
    print("Launching DemoServer on http://0.0.0.0:8000  (Ctrl+C to stop)")
    print("=" * 54)
    if shutil.which("conda") and conda_env_exists(ENV_NAME):
        cmd = ["conda", "run", "--no-capture-output", "-n", ENV_NAME, "python", SERVER]
    else:
        print("  (conda env unavailable — launching with the current interpreter)")
        cmd = [sys.executable, SERVER]
    # cwd must be DemoServer so relative paths in server.py resolve.
    run(cmd, cwd=HERE, check=False)


def main():
    ap = argparse.ArgumentParser(description="Set up and/or launch the MMG Demo Server.")
    ap.add_argument("--run", action="store_true", help="launch the server after setup")
    ap.add_argument("--run-only", action="store_true", help="skip setup, just launch")
    args = ap.parse_args()

    print("=" * 54)
    print("MMG Outreach Demo Server — Setup")
    print("=" * 54)

    if args.run_only:
        launch()
        return

    py_ok = setup_python()
    fe_ok = setup_frontend()

    print("\n" + "=" * 54)
    print("Setup summary")
    print("=" * 54)
    print(f"  Python env ({ENV_NAME}) : {'OK' if py_ok else 'INCOMPLETE — see above'}")
    print(f"  Frontend build          : {'OK' if fe_ok else 'INCOMPLETE — see above'}")
    print("\nStart the server with:")
    print(f"  conda run -n {ENV_NAME} python DemoServer/server.py")
    print("  # or: python DemoServer/setup_demo_server.py --run-only")

    if args.run and py_ok and fe_ok:
        launch()
    elif args.run:
        print("\nSkipping launch because setup did not complete cleanly.")


if __name__ == "__main__":
    main()
