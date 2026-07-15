#!/usr/bin/env python3
"""
setup_emg.py — set up (and optionally run) the Spikerbox-EMG demo.

Creates a dedicated conda env `emg` (Python 3.12) and installs the
dependencies from requirements.txt.

  python setup_emg.py                 # create/update the env and install deps
  python setup_emg.py --list-devices  # list audio input devices (find the SpikerBox)
  python setup_emg.py --run           # set up (if needed) and launch main.py
  python setup_emg.py --run-only      # skip setup, just launch main.py

The SpikerBox streams over the audio input. main.py reads audio device index
`device_idx` (main.py line ~27, default 1) — use --list-devices to find the
right index and edit that line if it isn't 1.
"""

import argparse
import os
import shutil
import subprocess
import sys

ENV_NAME = "emg"
PY_VERSION = "3.12"

HERE = os.path.dirname(os.path.abspath(__file__))
REQS = os.path.join(HERE, "requirements.txt")
MAIN = os.path.join(HERE, "main.py")


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


def require_conda():
    if not shutil.which("conda"):
        print("ERROR: 'conda' not found on PATH. Install Miniconda/Anaconda first:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        return False
    return True


def conda_env_exists(name):
    try:
        out = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    for line in out.splitlines():
        parts = line.replace("*", " ").split()  # drop the active-env marker
        if parts and parts[0] == name:
            return True
    return False


def in_env(*cmd, check=True):
    """Run a command inside the emg conda env."""
    return run(["conda", "run", "--no-capture-output", "-n", ENV_NAME, *cmd], check=check)


def setup():
    print("=" * 54)
    print(f"Spikerbox-EMG — Setup  (conda env: {ENV_NAME})")
    print("=" * 54)
    if not require_conda():
        return False

    if conda_env_exists(ENV_NAME):
        print(f"  Env '{ENV_NAME}' already exists — reusing it.")
    else:
        print(f"  Creating env '{ENV_NAME}' (Python {PY_VERSION})...")
        if not run(["conda", "create", "-y", "-n", ENV_NAME, f"python={PY_VERSION}"]):
            return False

    print(f"  Installing requirements into '{ENV_NAME}'...")
    ok = in_env("pip", "install", "-r", REQS)
    if ok:
        print("\n  SUCCESS: dependencies installed.")
        print(f"  Run the demo with:  python setup_emg.py --run")
    else:
        print("\n  ERROR: dependency install failed — see the pip output above.")
        print("  If sounddevice fails, ensure PortAudio is available on your system.")
    return ok


def list_devices():
    if not require_conda() or not conda_env_exists(ENV_NAME):
        print("Env not set up yet — run 'python setup_emg.py' first.")
        return
    # sounddevice prints an indexed table; the '<' entries are inputs.
    in_env("python", "-c", "import sounddevice as sd; print(sd.query_devices())", check=False)


def launch():
    if not require_conda() or not conda_env_exists(ENV_NAME):
        print("Env not set up yet — run 'python setup_emg.py' first.")
        return
    print("=" * 54)
    print("Launching Spikerbox-EMG  (close the window or Ctrl+C to stop)")
    print("=" * 54)
    in_env("python", MAIN, check=False)


def main():
    ap = argparse.ArgumentParser(description="Set up and run the Spikerbox-EMG demo.")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--run", action="store_true", help="set up (if needed) then launch main.py")
    ap.add_argument("--run-only", action="store_true", help="skip setup, just launch main.py")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return
    if args.run_only:
        launch()
        return

    if setup() and args.run:
        launch()


if __name__ == "__main__":
    main()
