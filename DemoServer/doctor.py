#!/usr/bin/env python3
"""
doctor.py — troubleshoot and reset the MMG Demo Server.

The two recurring headaches with the demo server are (1) a leftover instance
holding the ports so a new one won't start, and (2) hitting the wrong
protocol on a port (http on the TLS port, or https on the redirect port),
which the browser reports as ERR_CONNECTION_RESET. This tool diagnoses both
and can free the ports for you.

  python doctor.py            # diagnose everything (read-only)
  python doctor.py --reset    # stop any running server / free ports 8443 + 8000
  python doctor.py --restart  # reset, then start a fresh server
  python doctor.py --http     # diagnose/plan for plain-HTTP mode (:8000 only)

Cross-platform (Linux/macOS/Windows). Uses psutil if available for reliable
process handling, otherwise falls back to OS commands.
"""

import argparse
import os
import socket
import ssl
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))

# Paths mirror server.py so we can check the same prerequisites.
GUI_DIR   = os.path.join(REPO, "Demos", "SSM Demo", "predict_gui", "TauriGUI")
VITE_DIST = os.path.join(GUI_DIR, "dist")
BONES     = os.path.join(GUI_DIR, "public", "bones.json")
RES_DIR   = os.path.join(REPO, "Demos", "SSM Demo", "predict_gui", "Resources")
ANTHRO    = os.path.join(RES_DIR, "anthro_data.csv")
SSM_MODEL = os.path.join(RES_DIR, "SSM_shape_model_103")
CERT_FILE = os.path.join(HERE, "certs", "cert.pem")
KEY_FILE  = os.path.join(HERE, "certs", "key.pem")
SERVER_PY = os.path.join(HERE, "server.py")
EMG_SCORES = os.path.join(HERE, "emg_scores.json")   # mirrors server.py EMG_SCORES_FILE

# Other served demos (static — no build step) + their key assets.
GUI_SRC   = os.path.join(GUI_DIR, "src")
GUI_PKG   = os.path.join(GUI_DIR, "package.json")
GUI_INDEX = os.path.join(GUI_DIR, "index.html")
DIST_INDEX = os.path.join(VITE_DIST, "index.html")
EMG_WEB   = os.path.join(REPO, "Demos", "Spikerbox-EMG", "web", "index.html")
SEG_WEB   = os.path.join(REPO, "Demos", "StrangeObjectSegmenter", "web", "index.html")
THREE_JS  = os.path.join(HERE, "resources", "vendor", "three", "three.module.js")

HTTPS_PORT = 8443
HTTP_PORT  = 8000

# mDNS name the server advertises (keep in sync with server.py MDNS_FQDN).
MDNS_FQDN = "mmg-demo.local"


# ---- demo build/health status -------------------------------------------
def _newest_mtime(root, prune=()):
    """Newest file mtime under `root` (a file or dir), skipping `prune` dirs."""
    if os.path.isfile(root):
        try:
            return os.path.getmtime(root)
        except OSError:
            return 0.0
    newest = 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in prune]
        for f in filenames:
            try:
                m = os.path.getmtime(os.path.join(dirpath, f))
            except OSError:
                continue
            if m > newest:
                newest = m
    return newest


def demo_status():
    """Describe each served demo and whether it needs (re)building.

    Returns a list of dicts: {name, route, status, detail}. `status` is one of
    ready | stale | build | degraded | missing (worst-case per demo). Purely
    filesystem-based, so it's meaningful whether or not the server is running.
    """
    demos = []

    # --- Interactive Biomechanics (SSM) — the one demo with a build step ---
    if not os.path.isdir(VITE_DIST):
        ssm = ("build", "frontend not built — run setup_demo_server.py")
    else:
        built = _newest_mtime(DIST_INDEX)
        src = max(_newest_mtime(GUI_SRC, prune=("node_modules", "dist", ".git")),
                  _newest_mtime(GUI_PKG), _newest_mtime(GUI_INDEX))
        if src > built > 0:
            ssm = ("stale", "source changed since last build — rebuild")
        elif not (os.path.isdir(SSM_MODEL) and os.path.exists(ANTHRO)):
            ssm = ("degraded", "built, but model/data missing — predictions fail")
        else:
            ssm = ("ready", "built · model loaded")
    demos.append({"name": "Interactive Biomechanics", "route": "/ssm/",
                  "status": ssm[0], "detail": ssm[1]})

    # --- Muscles in Control (EMG) — static, no build -----------------------
    if os.path.exists(EMG_WEB):
        emg = ("ready", "static · no build needed")
    else:
        emg = ("missing", "web/index.html not found")
    demos.append({"name": "Muscles in Control", "route": "/emg/",
                  "status": emg[0], "detail": emg[1]})

    # --- Object Segmenter — static, needs vendored three.js for 3D --
    if not os.path.exists(SEG_WEB):
        seg = ("missing", "web/index.html not found")
    elif not os.path.exists(THREE_JS):
        seg = ("degraded", "three.js vendor missing — 3D reveal disabled")
    else:
        seg = ("ready", "static · no build needed")
    demos.append({"name": "Object Segmenter", "route": "/segment/",
                  "status": seg[0], "detail": seg[1]})

    return demos

# ---- pretty output -------------------------------------------------------
def line(label, status, note=""):
    tag = {"OK": "OK  ", "FAIL": "FAIL", "WARN": "WARN", "INFO": "INFO"}[status]
    print(f"  [{tag}] {label}" + (f"  - {note}" if note else ""))

def head(t):
    print("-" * 54); print(t); print("-" * 54)


# ---- networking ----------------------------------------------------------
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()

def port_free(port):
    # Set SO_REUSEADDR to match how the real server binds: this still fails if a
    # live process is listening (a genuine conflict) but succeeds over lingering
    # TIME_WAIT sockets, so we don't false-alarm right after a reset.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", port)); return True
    except OSError:
        return False
    finally:
        s.close()


# ---- process / port discovery (psutil first, OS fallback) ----------------
def _listeners_psutil(ports):
    """{port: [(pid, name, cmdline)]} via psutil, or None if psutil absent or
    the connection scan itself fails (falls back to the OS-tool path below)."""
    try:
        import psutil
    except ImportError:
        return None
    found = {p: [] for p in ports}
    try:
        conns = psutil.net_connections(kind="inet")
    except Exception:
        # Can raise transiently (e.g. AccessDenied) — most often right as a
        # process is exiting. Fall back rather than let this crash the whole
        # status poll (StatusWorker.run) and freeze the dashboard's UI updates.
        return None
    for c in conns:
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port in ports and c.pid:
            try:
                pr = psutil.Process(c.pid)
                found[c.laddr.port].append((c.pid, pr.name(), " ".join(pr.cmdline())))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                found[c.laddr.port].append((c.pid, "?", "?"))
    return found

def _listeners_os(ports):
    """Best-effort {port: [(pid, name, cmdline)]} using OS tools."""
    found = {p: [] for p in ports}
    if os.name == "nt":
        try:
            out = subprocess.run(["netstat", "-ano", "-p", "tcp"],
                                 capture_output=True, text=True).stdout
            for ln in out.splitlines():
                parts = ln.split()
                if len(parts) >= 5 and parts[3] == "LISTENING":
                    for p in ports:
                        if parts[1].endswith(f":{p}"):
                            found[p].append((int(parts[4]), "python?", "(netstat)"))
        except FileNotFoundError:
            pass
    else:
        # Linux: `ss` is part of iproute2 and almost always present. macOS has
        # no `ss`, so fall through to `lsof` (which ships with macOS).
        import re
        try:
            out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
            for ln in out.splitlines():
                for p in ports:
                    if f":{p} " in ln:
                        m = re.search(r"pid=(\d+)", ln)
                        nm = re.search(r'"([^"]+)"', ln)
                        if m:
                            found[p].append((int(m.group(1)), nm.group(1) if nm else "?", "(ss)"))
            if any(found.values()):
                return found
        except FileNotFoundError:
            pass
        for p in ports:
            try:
                out = subprocess.run(["lsof", f"-ti:{p}", "-sTCP:LISTEN"],
                                     capture_output=True, text=True).stdout
                for pid in out.split():
                    found[p].append((int(pid), "?", "(lsof)"))
            except FileNotFoundError:
                pass
    return found

def find_listeners(ports):
    return _listeners_psutil(ports) or _listeners_os(ports)

def is_demo_server(cmdline):
    """True if a listener's `cmdline` looks like our own server.py, rather than
    some unrelated process that merely grabbed port 8443/8000. Unknown/placeholder
    cmdlines (the OS-tool fallback yields "(ss)"/"(lsof)"/"?" instead of a real
    command line) are treated as ours, so we never hide a server we simply can't
    introspect."""
    if not cmdline or cmdline == "?" or cmdline.startswith("("):
        return True
    return "server.py" in cmdline.replace("\\", "/").lower()

def find_server_listeners(ports):
    """`find_listeners` filtered to just the MMG demo server — drops PIDs whose
    cmdline shows they're a different service holding the port."""
    listeners = find_listeners(ports)
    return {p: [hit for hit in hits if is_demo_server(hit[2])]
            for p, hits in listeners.items()}

def kill_pid(pid):
    try:
        import psutil
        pr = psutil.Process(pid)
        pr.terminate()
        try:
            pr.wait(timeout=3)
        except psutil.TimeoutExpired:
            pr.kill()
        return True
    except ImportError:
        pass
    except Exception:
        return False
    # No psutil: OS command.
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True)
        return True
    except Exception:
        return False


# ---- protocol probe ------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # Don't follow redirects — we want to *see* the 302 from the :8000 redirector,
    # not silently land on the 200 at :8443.
    def redirect_request(self, *a, **k):
        return None


def _get(url, timeout=4, method="GET"):
    """Return (status_code, detail). status_code None on transport error."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.HTTPSHandler(context=ctx))
    try:
        req = urllib.request.Request(url, method=method)
        with opener.open(req, timeout=timeout) as r:
            return r.status, ""
    except urllib.error.HTTPError as e:
        return e.code, ""            # a 302 lands here and counts as "spoke correctly"
    except Exception as e:
        return None, type(e).__name__ + ": " + str(e)


# ---- commands ------------------------------------------------------------
def diagnose(want_https=True):
    ip = lan_ip()

    head("Prerequisites")
    line("SSM frontend built", "OK" if os.path.isdir(VITE_DIST) else "FAIL",
         "" if os.path.isdir(VITE_DIST) else "run setup_demo_server.py (vite build)")
    line("SSM model + data", "OK" if os.path.isdir(SSM_MODEL) and os.path.exists(ANTHRO) else "FAIL")
    line("bones.json", "OK" if os.path.exists(BONES) else "WARN",
         "" if os.path.exists(BONES) else "auto-generated on first server start")
    if want_https:
        line("TLS cert", "OK" if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE) else "WARN",
             "" if os.path.exists(CERT_FILE) else "auto-generated on first start (self-signed)")

    head("Ports")
    listeners = find_listeners([HTTPS_PORT, HTTP_PORT])
    running = False
    for port in ([HTTPS_PORT, HTTP_PORT] if want_https else [HTTP_PORT]):
        holders = listeners.get(port, [])
        if not holders:
            line(f"port {port} free", "OK")
        else:
            running = True
            who = "; ".join(f"PID {pid} ({name})" for pid, name, _ in holders)
            line(f"port {port} in use", "WARN", who + "  → `doctor.py --reset` to free")

    head("Protocol check" if running else "Protocol check (server not running)")
    if not running:
        line("server", "INFO", "not running — start it, then re-run doctor to probe")
    else:
        if want_https:
            code, _ = _get(f"https://localhost:{HTTPS_PORT}/")
            line(f"https://…:{HTTPS_PORT}", "OK" if code == 200 else "FAIL",
                 f"HTTP {code}" if code else "no valid HTTPS response")
            code, _ = _get(f"http://localhost:{HTTP_PORT}/")
            line(f"http://…:{HTTP_PORT} (redirect)", "OK" if code in (301, 302) else "WARN",
                 f"HTTP {code}" if code else "redirector not answering")
        else:
            code, _ = _get(f"http://localhost:{HTTP_PORT}/")
            line(f"http://…:{HTTP_PORT}", "OK" if code == 200 else "FAIL", f"HTTP {code}" if code else "")

    head("Open the demo at")
    if want_https:
        print(f"  This device : https://localhost:{HTTPS_PORT}")
        print(f"  Tablets     : https://{ip}:{HTTPS_PORT}   (or http://{ip}:{HTTP_PORT} — redirects)")
        print()
        print("  DO NOT use  : http://…:%d  or  https://…:%d   (both → ERR_CONNECTION_RESET)"
              % (HTTPS_PORT, HTTP_PORT))
    else:
        print(f"  This device : http://localhost:{HTTP_PORT}")
        print(f"  Tablets     : http://{ip}:{HTTP_PORT}   (mic blocked without HTTPS)")
    print("-" * 54)


def reset(want_https=True):
    head("Reset — freeing the demo server ports")
    ports = [HTTPS_PORT, HTTP_PORT] if want_https else [HTTP_PORT]
    listeners = find_listeners(ports)
    pids = {pid for hs in listeners.values() for pid, _, _ in hs}
    if not pids:
        line("ports", "OK", "nothing to stop — already free")
    else:
        for pid in sorted(pids):
            ok = kill_pid(pid)
            line(f"stop PID {pid}", "OK" if ok else "FAIL",
                 "stopped" if ok else "could not kill (try running as the same user/admin)")
    # Verify
    import time; time.sleep(1)
    all_free = all(port_free(p) for p in ports)
    line("ports free now", "OK" if all_free else "FAIL",
         "" if all_free else "something is still holding a port")
    print("-" * 54)
    return all_free


def restart(want_https):
    if not reset(want_https):
        print("Not restarting — ports could not be freed.")
        return
    head("Starting a fresh server")
    cmd = [sys.executable, SERVER_PY] + ([] if want_https else ["--http"])
    print("  $ " + " ".join(cmd))
    os.execv(sys.executable, cmd)   # replace this process with the server


def main():
    ap = argparse.ArgumentParser(description="Troubleshoot and reset the MMG Demo Server.")
    ap.add_argument("--reset", action="store_true", help="stop any running server / free the ports")
    ap.add_argument("--restart", action="store_true", help="reset, then start a fresh server")
    ap.add_argument("--http", action="store_true", help="target plain-HTTP mode (:8000 only)")
    args = ap.parse_args()
    want_https = not args.http

    print("=" * 54)
    print("MMG Demo Server — Doctor")
    print("=" * 54)

    if args.restart:
        restart(want_https)
    elif args.reset:
        reset(want_https)
    else:
        diagnose(want_https)


if __name__ == "__main__":
    main()
