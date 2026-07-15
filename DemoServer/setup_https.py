#!/usr/bin/env python3
"""
setup_https.py — give the DemoServer a *trusted* HTTPS cert via mkcert.

By default `server.py` serves HTTPS with an auto-generated self-signed cert,
which makes browsers show a "not secure" warning. This helper uses mkcert to
create a cert signed by a local CA that your machine trusts, so there is **no
warning on this PC** (and none on Android after a one-time tap-through). For
iPads, install mkcert's root CA on the device once — this script prints where
to find it.

  python setup_https.py                  # trust + cert for localhost + this LAN IP
  python setup_https.py demo.local 10.0.0.5   # also include extra hostnames/IPs

Writes certs/cert.pem and certs/key.pem, which server.py picks up automatically
on its next start. mkcert is auto-installed via conda-forge if conda is around.
"""

import os
import sys
import shutil
import socket
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CERT_DIR = os.path.join(HERE, "certs")
CERT_FILE = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE = os.path.join(CERT_DIR, "key.pem")


def lan_ip():
    """Best-effort primary LAN IP (so tablets on the same network can connect)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # no packets sent; just picks the route
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def run(cmd):
    print("\n$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        print(f"  ERROR: {e}")
        return False


def find_mkcert():
    """Return a path to mkcert, installing it via conda-forge if needed."""
    exe = shutil.which("mkcert")
    if exe:
        return exe
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        print("mkcert not found on PATH — installing via conda-forge...")
        run([conda, "install", "-y", "-c", "conda-forge", "mkcert"])
        exe = shutil.which("mkcert")
        if exe:
            return exe
    return None


def install_hint():
    print("\nCould not find or install mkcert. Install it, then re-run this script:")
    if sys.platform.startswith("win"):
        print("  choco install mkcert       (or: scoop install mkcert)")
    elif sys.platform == "darwin":
        print("  brew install mkcert")
    else:
        print("  see https://github.com/FiloSottile/mkcert#installation")
    print("  or, with conda: conda install -c conda-forge mkcert")


def main():
    extra = sys.argv[1:]
    print("=" * 54)
    print("DemoServer HTTPS — trusted cert via mkcert")
    print("=" * 54)

    mkcert = find_mkcert()
    if not mkcert:
        install_hint()
        return 1

    # 1) Install the local CA into the system trust store (no-op if already done).
    #    On Windows this uses the current user's store — no admin needed normally.
    if not run([mkcert, "-install"]):
        print("  (If this failed, you may need admin rights, or an endpoint")
        print("   security agent may be blocking trust-store changes.)")
        return 1

    # 2) Generate the leaf cert for localhost, 127.0.0.1, the LAN IP, plus extras.
    os.makedirs(CERT_DIR, exist_ok=True)
    hosts = ["localhost", "127.0.0.1", lan_ip()] + extra
    seen = set()
    hosts = [h for h in hosts if not (h in seen or seen.add(h))]
    if not run([mkcert, "-cert-file", CERT_FILE, "-key-file", KEY_FILE, *hosts]):
        return 1

    # 3) Report where the root CA lives (for installing on tablets).
    caroot = ""
    try:
        caroot = subprocess.run(
            [mkcert, "-CAROOT"], capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        pass

    print("\n" + "=" * 54)
    print("Done — server.py will use these on its next start:")
    print(f"  Cert : {CERT_FILE}")
    print(f"  Key  : {KEY_FILE}")
    print(f"  Hosts: {', '.join(hosts)}")
    if caroot:
        rootca = os.path.join(caroot, "rootCA.pem")
        print("\nFor iPad / Android tablets, install the root CA on each device once:")
        print(f"  {rootca}")
        print("  iOS: AirDrop/email rootCA.pem → install the profile → Settings →")
        print("       General → About → Certificate Trust Settings → enable it.")
    print("\nRestart the server (Ctrl+C, then relaunch) to load the new cert.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
