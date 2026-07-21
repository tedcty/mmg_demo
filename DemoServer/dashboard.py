#!/usr/bin/env python3
"""
dashboard.py — Material admin-style control panel for the MMG Demo Server.

A standalone operator app laid out like a monitoring dashboard: a dark sidebar
with page navigation, KPI stat cards, a live CPU chart, diagnostics, protocol
health, an access panel and a console — with server control and optional
auto-refresh. All the underlying logic is reused from doctor.py.

  conda run -n demo python DemoServer/dashboard.py

Built on PySide6 (already in the demo env via ptb_mmg) + qt-material, with a
custom stylesheet layered on top. Run it in the `demo` conda env so it can
launch the server with the right interpreter.
"""

import io
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

from PySide6 import QtCore, QtGui, QtWidgets

import doctor  # same folder — reuse all status/reset logic

# ---- palette (UoA 2025 brand) -------------------------------------------
AZURE  = "#1f2bd4"
INK    = "#0c0c48"
NAVY   = "#0c0c48"   # sidebar
GREEN  = "#12b57f"
RED    = "#e5484d"
LAVA   = "#f0501e"   # molten orange-red — the live "Stop" button
LAVA_DK = "#d5400f"
AMBER  = "#e0a100"
PURPLE = "#7c4dff"
GREY   = "#8a8f98"
LABEL  = "#3f4652"   # darker slate for section titles / field labels (readable)
BG     = "#eef1f6"
CARD   = "#ffffff"
BORDER = "#e6e9f0"

# Type scale is deliberately generous — the panel is used on 13–16" portable
# outreach laptops, so labels/values are sized to read at arm's length.
QSS = f"""
QWidget#root {{ background: {BG}; }}
QToolTip {{ font-size: 13px; }}

/* Sidebar */
QFrame#sidebar {{ background: {NAVY}; }}
QLabel#brand {{ color: white; font-size: 18px; font-weight: 800; }}
QLabel#brandSub {{ color: rgba(255,255,255,0.55); font-size: 11px; font-weight: 600; }}
QPushButton#nav {{ color: #c7cbe8; background: transparent; border: none;
                   text-align: left; padding: 13px 16px; border-radius: 10px;
                   font-size: 15px; font-weight: 600; }}
QPushButton#nav:hover {{ background: rgba(255,255,255,0.07); color: white; }}
QPushButton#nav:checked {{ background: {AZURE}; color: white; }}
QLabel#sideFoot {{ color: rgba(255,255,255,0.4); font-size: 11px; }}

/* Header */
QLabel#h1 {{ color: {INK}; font-size: 23px; font-weight: 800; }}
QLabel#chip {{ font-size: 14px; font-weight: 700; border-radius: 14px; padding: 6px 15px; }}

/* Cards */
QFrame#stat, QGroupBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#statValue {{ font-size: 23px; font-weight: 800; color: {INK}; }}
QLabel#statSub {{ color: {LABEL}; font-size: 14px; font-weight: 800; }}
QLabel#statCaption {{ color: {GREY}; font-size: 11px; font-weight: 700; }}
QGroupBox {{ margin-top: 28px; padding: 16px; font-weight: 700; color: {INK}; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left;
                    left: 16px; top: 4px; padding: 0 4px;
                    color: {LABEL}; font-size: 16px; font-weight: 800; }}
QLabel#diag {{ font-size: 15px; color: {INK}; }}
QLabel#url {{ font-size: 14px; color: {INK}; }}

/* Demo list items */
QFrame#demoItem {{ background: #f7f9fc; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#demoName {{ color: {INK}; font-size: 15px; font-weight: 800; }}
QLabel#demoDetail {{ color: {GREY}; font-size: 13px; font-weight: 600; }}
QLabel#demoChip {{ font-size: 11px; font-weight: 800; border-radius: 10px; padding: 3px 10px; }}

/* Buttons */
QPushButton {{ background: #f1f4f9; color: {INK}; border: 1px solid #dde3ec;
               border-radius: 10px; padding: 11px 18px; font-size: 14px; font-weight: 700; }}
QPushButton:hover {{ background: #e7ecf5; }}
QPushButton:disabled {{ color: #b3bac6; background: #f4f6fa; }}
QPushButton#primary {{ background: {AZURE}; color: white; border: none; }}
QPushButton#primary:hover {{ background: #1a24b0; }}
QPushButton#primary:disabled {{ background: #b9bdec; }}
QPushButton#danger {{ background: #fdecea; color: {RED}; border: 1px solid #f6c9c7; }}
QPushButton#danger:hover {{ background: #fadedb; }}

/* Server control buttons — large, easy-to-hit targets */
QPushButton[ctl="true"] {{ font-size: 16px; font-weight: 800;
                           padding: 15px 24px; border-radius: 12px; }}

/* The big Start/Stop toggle in the Diagnostics column */
QPushButton[power="true"] {{ font-size: 21px; font-weight: 800;
                             padding: 16px; border-radius: 14px; }}
/* "Stop" state — molten lava so an active/running server is unmistakable */
QPushButton#lava {{ background: {LAVA}; color: white; border: none; }}
QPushButton#lava:hover {{ background: {LAVA_DK}; }}
QPushButton#lava:disabled {{ background: #f4b6a3; color: #fdeee8; }}

QPlainTextEdit#console, QPlainTextEdit#docout {{
    background: #0f1424; color: #c7d2e0; border: none; border-radius: 12px;
    padding: 8px; font-family: monospace; font-size: 13px; }}
QCheckBox {{ color: {INK}; font-size: 14px; font-weight: 600; }}
QCheckBox::indicator {{ width: 18px; height: 18px; }}
QSpinBox {{ font-size: 14px; padding: 4px 6px; }}

/* Startup progress (indeterminate) */
QLabel#startupLbl {{ color: {AZURE}; font-size: 13px; font-weight: 700; }}
QProgressBar#startup {{ background: #e7ecf5; border: none; border-radius: 6px;
                        min-height: 9px; max-height: 9px; }}
QProgressBar#startup::chunk {{ background: {AZURE}; border-radius: 6px; }}

/* Tablet access banner */
QFrame#access {{ background: {INK}; border-radius: 16px; }}
QLabel#accessCap {{ color: rgba(255,255,255,0.55); font-size: 12px; font-weight: 800; }}
QLabel#accessUrl {{ color: white; font-size: 29px; font-weight: 800; }}
QLabel#accessHint {{ color: rgba(255,255,255,0.6); font-size: 12px; font-weight: 600; }}
QPushButton#accessCopy {{ background: rgba(255,255,255,0.14); color: white; border: none;
                          border-radius: 10px; padding: 11px 18px; font-weight: 700; }}
QPushButton#accessCopy:hover {{ background: rgba(255,255,255,0.24); }}
QLabel#qrTile {{ background: #ffffff; border-radius: 8px; }}
QLabel#alert {{ background: #fdecea; color: {RED}; border: 1px solid #f6c9c7;
                border-radius: 10px; padding: 11px 15px; font-size: 14px; font-weight: 700; }}

/* QR / trust panels on the Access page */
QLabel#qrCard {{ background: #ffffff; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#qrCaption {{ color: {INK}; font-size: 15px; font-weight: 800; }}
QLabel#qrSub {{ color: {GREY}; font-size: 13px; font-weight: 600; }}
"""


# ---- little vector icons (reliable across platforms) --------------------
def _draw_shape(p, kind, size, color):
    p.setPen(QtGui.QPen(QtGui.QColor(color), 2.0))
    p.setBrush(QtGui.QColor(color))
    c = size / 2
    if kind == "status":
        p.drawEllipse(QtCore.QPointF(c, c), size * 0.16, size * 0.16)
    elif kind == "clock":
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(c, c), size * 0.24, size * 0.24)
        p.drawLine(QtCore.QPointF(c, c), QtCore.QPointF(c, c - size * 0.16))
        p.drawLine(QtCore.QPointF(c, c), QtCore.QPointF(c + size * 0.12, c))
    elif kind == "users":
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(c - size * 0.13, c - size * 0.08), size * 0.11, size * 0.11)
        p.drawEllipse(QtCore.QPointF(c + size * 0.13, c - size * 0.08), size * 0.11, size * 0.11)
        p.drawRoundedRect(QtCore.QRectF(c - size * 0.26, c + size * 0.05, size * 0.52, size * 0.18),
                          size * 0.08, size * 0.08)
    elif kind == "cpu":
        p.setPen(QtCore.Qt.NoPen)
        for i, h in enumerate((0.16, 0.30, 0.22)):
            x = c - size * 0.22 + i * size * 0.18
            p.drawRoundedRect(QtCore.QRectF(x, c + size * 0.14 - size * h, size * 0.1, size * h), 1.5, 1.5)
    elif kind == "grid":  # dashboard
        p.setPen(QtCore.Qt.NoPen)
        for dx in (-0.2, 0.05):
            for dy in (-0.2, 0.05):
                p.drawRoundedRect(QtCore.QRectF(c + dx * size, c + dy * size, size * 0.15, size * 0.15), 2, 2)
    elif kind == "term":  # console
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawRoundedRect(QtCore.QRectF(c - size * 0.26, c - size * 0.2, size * 0.52, size * 0.4), 3, 3)
        p.drawLine(QtCore.QPointF(c - size * 0.14, c - size * 0.05), QtCore.QPointF(c - size * 0.04, c + size * 0.03))
        p.drawLine(QtCore.QPointF(c - size * 0.04, c + size * 0.03), QtCore.QPointF(c - size * 0.14, c + size * 0.11))
    elif kind == "link":  # access
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(c - size * 0.12, c - size * 0.12), size * 0.12, size * 0.12)
        p.drawEllipse(QtCore.QPointF(c + size * 0.12, c + size * 0.12), size * 0.12, size * 0.12)
    elif kind == "cross":  # doctor / medical
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(QtCore.QRectF(c - size * 0.07, c - size * 0.24, size * 0.14, size * 0.48), 2, 2)
        p.drawRoundedRect(QtCore.QRectF(c - size * 0.24, c - size * 0.07, size * 0.48, size * 0.14), 2, 2)
    elif kind == "play":  # start
        p.setPen(QtCore.Qt.NoPen)
        p.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(c - size * 0.16, c - size * 0.24),
            QtCore.QPointF(c - size * 0.16, c + size * 0.24),
            QtCore.QPointF(c + size * 0.24, c)]))
    elif kind == "stopsq":  # stop
        p.setPen(QtCore.Qt.NoPen)
        p.drawRoundedRect(QtCore.QRectF(c - size * 0.2, c - size * 0.2, size * 0.4, size * 0.4), 2, 2)


def _pixmap(kind, color, size=40, bg=None):
    pm = QtGui.QPixmap(size, size); pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm); p.setRenderHint(QtGui.QPainter.Antialiasing)
    if bg:
        p.setPen(QtCore.Qt.NoPen); p.setBrush(QtGui.QColor(bg))
        p.drawRoundedRect(0, 0, size, size, size * 0.28, size * 0.28)
    _draw_shape(p, kind, size, color)
    p.end()
    return pm


def _dot(color):
    return f'<span style="color:{color};font-size:15px">●</span>'


def _qr_pixmap(data, size, dark=INK):
    """Render `data` as a QR-code QPixmap sized to `size` px, or None if the
    `segno` package isn't available. Pure-Python, no network/Pillow needed."""
    try:
        import segno
    except Exception:
        return None
    qr = segno.make(data, error="m")
    buf = io.BytesIO()
    qr.save(buf, kind="png", scale=10, border=2, dark=dark, light="#ffffff")
    pm = QtGui.QPixmap()
    pm.loadFromData(buf.getvalue(), "PNG")
    # FastTransformation keeps the modules crisp (smooth-scaling blurs QR edges).
    return pm.scaled(size, size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)


# ---- GPU telemetry (best effort: pynvml → nvidia-smi → none) -------------
_GPU = {"mode": "auto", "handle": None}


def _gpu_stats():
    """Return (util_pct, mem_used_mb, mem_total_mb) or None if no GPU."""
    mode = _GPU["mode"]
    if mode in ("auto", "nvml"):
        try:
            import pynvml
            if _GPU["handle"] is None:
                pynvml.nvmlInit()
                _GPU["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(_GPU["handle"]).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU["handle"])
            _GPU["mode"] = "nvml"
            return float(util), mem.used / 1048576, mem.total / 1048576
        except Exception:
            _GPU["handle"] = None
            if mode == "nvml":
                return None
    if mode in ("auto", "smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            row = out.stdout.strip().splitlines()[0]
            util, used, total = [float(x) for x in row.split(",")]
            _GPU["mode"] = "smi"
            return util, used, total
        except Exception:
            if mode == "smi":
                return None
    _GPU["mode"] = "none"
    return None


def _gpu_proc_util(pids):
    """Best-effort GPU SM-utilization (%) summed over `pids`, via NVML's
    per-process sampler. Returns a float or None if unavailable (needs pynvml
    and a driver that supports process utilization)."""
    if _GPU.get("mode") not in ("nvml", "auto"):
        return None
    try:
        import time as _t
        import pynvml
        if _GPU.get("handle") is None:
            pynvml.nvmlInit()
            _GPU["handle"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        # samples from ~1s ago; take the latest sample per matching pid
        since = int((_t.time() - 1.0) * 1_000_000)
        latest = {}
        for smp in pynvml.nvmlDeviceGetProcessUtilization(_GPU["handle"], since):
            if smp.pid in pids:
                cur = latest.get(smp.pid)
                if cur is None or smp.timeStamp > cur[0]:
                    latest[smp.pid] = (smp.timeStamp, smp.smUtil)
        if not latest:
            return 0.0
        return float(min(sum(v[1] for v in latest.values()), 100.0))
    except Exception:
        return None


def _shadow(w, blur=24, alpha=26, dy=4):
    eff = QtWidgets.QGraphicsDropShadowEffect(w)
    eff.setBlurRadius(blur); eff.setXOffset(0); eff.setYOffset(dy)
    eff.setColor(QtGui.QColor(12, 12, 72, alpha))
    w.setGraphicsEffect(eff)


class StatusWorker(QtCore.QThread):
    done = QtCore.Signal(dict)

    def run(self):
        ports = [doctor.HTTPS_PORT, doctor.HTTP_PORT]
        # Only adopt our own server.py — not whatever else may hold the port.
        listeners = doctor.find_server_listeners(ports)
        pids = sorted({pid for hs in listeners.values() for pid, _, _ in hs})
        s = {
            "running": bool(pids), "pids": pids,
            "port_https": bool(listeners.get(doctor.HTTPS_PORT)),
            "port_http": bool(listeners.get(doctor.HTTP_PORT)),
            "dist": os.path.isdir(doctor.VITE_DIST),
            "model": os.path.isdir(doctor.SSM_MODEL) and os.path.exists(doctor.ANTHRO),
            "bones": os.path.exists(doctor.BONES),
            "cert": os.path.exists(doctor.CERT_FILE) and os.path.exists(doctor.KEY_FILE),
            "ip": doctor.lan_ip(),
            "https_code": None, "http_code": None,
            "create_time": None, "cpu": None, "mem_pct": None, "srv_mb": None,
            "srv_cpu": None, "srv_gpu": None,
            "clients": None, "gpu": None, "gpu_used": None, "gpu_total": None,
            "mdns_ok": None,
        }

        # Build the server process tree (listener PIDs + their children) — SSM
        # predictions run as *child* subprocesses and carry the real load, so
        # "server" load means the whole tree, not just the listener.
        srv_tree = []
        try:
            import psutil
            seen = set()
            for pid in pids:
                try:
                    proc = psutil.Process(pid)
                except Exception:
                    continue
                for q in [proc] + proc.children(recursive=True):
                    if q.pid not in seen:
                        seen.add(q.pid); srv_tree.append(q)
            for q in srv_tree:
                try:
                    q.cpu_percent(None)      # prime the per-process CPU counter
                except Exception:
                    pass
        except Exception:
            pass

        # Machine-wide CPU. The 0.2s window is shared with the per-process read
        # below, so both cover the same interval.
        try:
            import psutil
            s["cpu"] = psutil.cpu_percent(interval=0.2)
            s["mem_pct"] = psutil.virtual_memory().percent
            if srv_tree:
                ncpu = psutil.cpu_count() or 1
                cpu_sum = 0.0; rss = 0.0
                for q in srv_tree:
                    try:
                        cpu_sum += q.cpu_percent(None)
                        rss += q.memory_info().rss
                    except Exception:
                        pass
                s["srv_cpu"] = cpu_sum / ncpu     # % of total machine capacity
                s["srv_mb"] = rss / 1048576
        except Exception:
            pass

        gpu = _gpu_stats()
        if gpu is not None:
            s["gpu"], s["gpu_used"], s["gpu_total"] = gpu
        if srv_tree:
            s["srv_gpu"] = _gpu_proc_util({q.pid for q in srv_tree})

        try:
            s["demos"] = doctor.demo_status()
        except Exception:
            s["demos"] = []

        if s["running"]:
            s["https_code"], _ = doctor._get(f"https://localhost:{doctor.HTTPS_PORT}/", timeout=2)
            s["http_code"], _ = doctor._get(f"http://localhost:{doctor.HTTP_PORT}/", timeout=2)
            try:
                import psutil
                p = psutil.Process(pids[0])
                s["create_time"] = p.create_time()
                clients = 0
                for c in psutil.net_connections(kind="inet"):
                    if c.status == "ESTABLISHED" and c.laddr and c.laddr.port in ports:
                        clients += 1
                s["clients"] = clients
            except Exception:
                pass

            # mDNS reachability: a TCP connect to the advertised name both
            # resolves it (proving the name works from a separate process, a
            # good proxy for the tablets) and confirms the port is open.
            try:
                with socket.create_connection(
                        (doctor.MDNS_FQDN, doctor.HTTPS_PORT), timeout=1.5):
                    s["mdns_ok"] = True
            except Exception:
                s["mdns_ok"] = False
        self.done.emit(s)


class DoctorWorker(QtCore.QThread):
    done = QtCore.Signal(str)

    def run(self):
        import contextlib, io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                doctor.diagnose(True)
        except Exception as e:
            buf.write(f"\n[doctor error] {e}\n")
        self.done.emit(buf.getvalue())


class RebuildWorker(QtCore.QThread):
    """Runs `npx vite build --base /ssm/` in the TauriGUI dir, streaming output."""
    line = QtCore.Signal(str)
    done = QtCore.Signal(int)

    def __init__(self, gui_dir):
        super().__init__()
        self.gui_dir = gui_dir

    def run(self):
        cmd = ["npx", "vite", "build", "--base", "/ssm/"]
        use_shell = False
        # On Windows npx is a .cmd shim CreateProcess can't launch directly;
        # run it through the shell so PATHEXT resolves it (mirrors setup script).
        if os.name == "nt":
            resolved = shutil.which(cmd[0])
            if not resolved or not resolved.lower().endswith((".exe", ".com")):
                cmd = subprocess.list2cmdline(cmd)
                use_shell = True
        try:
            p = subprocess.Popen(
                cmd, cwd=self.gui_dir, shell=use_shell,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for ln in p.stdout:
                self.line.emit(ln.rstrip())
            p.wait()
            self.done.emit(p.returncode)
        except FileNotFoundError:
            self.line.emit("[rebuild] npx/Node not found — run setup_demo_server.py "
                           "to install Node into the env")
            self.done.emit(-1)
        except Exception as e:
            self.line.emit(f"[rebuild] error: {e}")
            self.done.emit(-1)


class Dashboard(QtWidgets.QWidget):
    _log_sig = QtCore.Signal(str)
    _exit_sig = QtCore.Signal(object, int)   # (proc, returncode) from the log pump

    NAV = [("grid", "Dashboard"), ("term", "Console"), ("link", "Access"), ("cross", "Doctor")]
    MAX_RESTARTS = 5          # keep-alive gives up after this many rapid crashes

    def __init__(self, autostart=False, keepalive=False):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle("MMG Demo Server — Control Panel")
        self.resize(1200, 760)
        self.setMinimumSize(880, 560)         # fits small 13" portable screens
        self.proc = None
        self._polling = False
        self._starting = False
        self._stopping = False
        self._restart_pending = False
        self._start_t0 = 0.0
        self._stop_t0 = 0.0
        self._url = f"https://localhost:{doctor.HTTPS_PORT}"
        self._tablet_url = self._url
        self._tablet_ip_url = self._url
        self._trust_url = f"http://localhost:{doctor.HTTP_PORT}/trust"
        self._rebuilding = False
        self._keepalive = keepalive           # auto-restart on unexpected exit
        self._crash_count = 0                 # consecutive keep-alive restarts
        self._last_restart_t = 0.0
        self._autostart = autostart
        self._autostart_done = False
        self._was_running = False             # previous observed running state
        self._adopted_pid = None              # PID of a server we didn't start
        self._exit_handled = False            # de-dupe crash handling per run
        self._last_rc = None                  # exit code from the pump, if known

        self._build_ui()
        if keepalive:
            self.keepalive.setChecked(True)
        self._log_sig.connect(self._append_log)
        self._exit_sig.connect(self._on_proc_exit)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self._toggle_auto(self.auto.isChecked())
        self._install_shortcuts()
        self.refresh()

    def _install_shortcuts(self):
        """Keyboard shortcuts for the common actions (shown in tooltips)."""
        def sc(seq, fn):
            QtGui.QShortcut(QtGui.QKeySequence(seq), self, activated=fn)
        sc("F5", self.refresh)
        # Respect button enabled-state so Start/Stop can't fire out of turn.
        # Start only when stopped, Stop only when running (ignore mid-transition).
        sc("Ctrl+S", lambda: self.power_btn.isEnabled() and not self._running_now()
           and self._on_start_clicked())
        sc("Ctrl+K", lambda: self.power_btn.isEnabled() and self._running_now()
           and self.stop_server())
        sc("Ctrl+R", self.restart_server)
        sc("Ctrl+Shift+R", self.rebuild_frontend)
        sc("Ctrl+D", self.run_doctor)
        sc("F11", self._toggle_fullscreen)
        sc("Esc", self._exit_fullscreen)
        for i in range(len(self.NAV)):
            sc(f"Alt+{i + 1}", lambda idx=i: self._go(idx))

    def _scrollable(self, page):
        """Wrap a page so it scrolls vertically when taller than the viewport
        (keeps cards at full size instead of squashing them until they overlap)."""
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QtWidgets.QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Keep the app's background showing through the viewport + page (the
        # cards keep their own white via #stat / #qrCard styling).
        sa.setStyleSheet("QScrollArea { background: transparent; }"
                         " QScrollArea > QWidget > QWidget { background: transparent; }")
        sa.setWidget(page)
        return sa

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._sync_fs_btn()

    def _exit_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self._sync_fs_btn()

    def _sync_fs_btn(self):
        self.fs_btn.setText("Exit full screen" if self.isFullScreen() else "Full screen")

    # ---- UI --------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        outer.addWidget(self._sidebar())

        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(22, 18, 22, 18); right.setSpacing(16)

        # Header
        hdr = QtWidgets.QHBoxLayout()
        self.h1 = QtWidgets.QLabel("Dashboard"); self.h1.setObjectName("h1")
        self.chip = QtWidgets.QLabel("…"); self.chip.setObjectName("chip")
        hbtn = QtWidgets.QPushButton("Refresh"); hbtn.clicked.connect(self.refresh)
        hbtn.setToolTip("Refresh status now (F5)")
        self.fs_btn = QtWidgets.QPushButton("Full screen")
        self.fs_btn.setToolTip("Toggle full screen (F11 · Esc to exit)")
        self.fs_btn.clicked.connect(self._toggle_fullscreen)
        hdr.addWidget(self.h1); hdr.addStretch(1)
        hdr.addWidget(self.chip); hdr.addWidget(self.fs_btn); hdr.addWidget(hbtn)
        right.addLayout(hdr)

        # Dismissable alert banner (e.g. unexpected server exit) — hidden by default.
        self.alert = QtWidgets.QLabel(""); self.alert.setObjectName("alert")
        self.alert.setWordWrap(True); self.alert.setVisible(False)
        self.alert.setCursor(QtCore.Qt.PointingHandCursor)
        self.alert.setToolTip("Click to dismiss")
        self.alert.installEventFilter(self)
        right.addWidget(self.alert)

        # Pages — each wrapped in a scroll area so a page taller than the window
        # scrolls instead of compressing its cards (which would overlap).
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._scrollable(self._page_dashboard()))
        self.stack.addWidget(self._scrollable(self._page_console()))
        self.stack.addWidget(self._scrollable(self._page_access()))
        self.stack.addWidget(self._scrollable(self._page_doctor()))
        right.addWidget(self.stack, 1)

        # Persistent control bar
        right.addLayout(self._control_bar())

        rw = QtWidgets.QWidget(); rw.setLayout(right)
        outer.addWidget(rw, 1)

    def _sidebar(self):
        bar = QtWidgets.QFrame(); bar.setObjectName("sidebar"); bar.setFixedWidth(212)
        v = QtWidgets.QVBoxLayout(bar); v.setContentsMargins(16, 20, 16, 16); v.setSpacing(6)
        logo = QtWidgets.QHBoxLayout()
        dot = QtWidgets.QLabel(); dot.setPixmap(_pixmap("cross", "white", 34, bg=AZURE))
        tv = QtWidgets.QVBoxLayout(); tv.setSpacing(0)
        b = QtWidgets.QLabel("MMG Server"); b.setObjectName("brand")
        bs = QtWidgets.QLabel("OUTREACH HUB"); bs.setObjectName("brandSub")
        tv.addWidget(b); tv.addWidget(bs)
        logo.addWidget(dot); logo.addSpacing(8); logo.addLayout(tv); logo.addStretch(1)
        v.addLayout(logo); v.addSpacing(18)

        self.nav_group = QtWidgets.QButtonGroup(self)
        for i, (icon, name) in enumerate(self.NAV):
            btn = QtWidgets.QPushButton(name); btn.setObjectName("nav")
            btn.setCheckable(True); btn.setIcon(QtGui.QIcon(_pixmap(icon, "#c7cbe8", 22)))
            btn.setIconSize(QtCore.QSize(18, 18))
            btn.setToolTip(f"{name} (Alt+{i + 1})")
            btn.clicked.connect(lambda _=False, idx=i: self._go(idx))
            self.nav_group.addButton(btn, i)
            v.addWidget(btn)
        self.nav_group.button(0).setChecked(True)
        v.addStretch(1)
        foot = QtWidgets.QLabel("Waipapa Taumata Rau\nUniversity of Auckland"); foot.setObjectName("sideFoot")
        v.addWidget(foot)
        return bar

    def _go(self, idx):
        self.stack.setCurrentIndex(idx)
        self.h1.setText(self.NAV[idx][1])
        self.nav_group.button(idx).setChecked(True)   # keep highlight in sync

    def _stat_card(self, icon, caption, color):
        card = QtWidgets.QFrame(); card.setObjectName("stat"); _shadow(card)
        # Fill the row equally; fixed height keeps the single row compact.
        card.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        card.setMinimumWidth(150); card.setFixedHeight(86)
        h = QtWidgets.QHBoxLayout(card); h.setContentsMargins(14, 12, 14, 12); h.setSpacing(12)
        ic = QtWidgets.QLabel(); ic.setPixmap(_pixmap(icon, "white", 44, bg=color))
        ic.setFixedSize(44, 44)
        tv = QtWidgets.QVBoxLayout(); tv.setSpacing(2)
        tv.addStretch(1)
        val = QtWidgets.QLabel("—"); val.setObjectName("statValue")
        sub = QtWidgets.QLabel(""); sub.setObjectName("statSub"); sub.hide()
        cap = QtWidgets.QLabel(caption); cap.setObjectName("statCaption")
        tv.addWidget(val); tv.addWidget(sub); tv.addWidget(cap)
        tv.addStretch(1)
        h.addWidget(ic, 0, QtCore.Qt.AlignVCenter); h.addLayout(tv); h.addStretch(1)
        return card, val, sub, ic

    def _page_dashboard(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(12)

        v.addWidget(self._build_access_banner())

        cards = QtWidgets.QHBoxLayout(); cards.setSpacing(14)
        self.c_status, self.k_status, _, self.status_icon = self._stat_card("status", "STATUS", GREY)
        cards.addWidget(self.c_status, 1)     # equal stretch → equal-width cards
        _, self.k_uptime, _ = self._stat_card_add(cards, "clock", "UPTIME", AZURE)
        _, self.k_clients, _ = self._stat_card_add(cards, "users", "CLIENTS", GREEN)
        _, self.k_cpu, self.k_cpu_sub = self._stat_card_add(cards, "cpu", "CPU LOAD", PURPLE)
        _, self.k_gpu, self.k_gpu_sub = self._stat_card_add(cards, "cpu", "GPU LOAD", GREEN)
        # CPU/GPU cards carry a second "server …" line under the machine total.
        for sub in (self.k_cpu_sub, self.k_gpu_sub):
            sub.setText("server —"); sub.show()
        v.addLayout(cards)

        # Demos + diagnostics side by side
        midrow = QtWidgets.QHBoxLayout(); midrow.setSpacing(16)
        midrow.addWidget(self._build_demos_box(), 3)

        diag = QtWidgets.QGroupBox("Diagnostics"); _shadow(diag)
        dl = QtWidgets.QVBoxLayout(diag); dl.setSpacing(10)
        self.ports_lbl = self._diag_row(dl, "Ports")
        self.health_lbl = self._diag_row(dl, "Health")
        self.mdns_lbl = self._diag_row(dl, "Name")
        self.deps_lbl = self._diag_row(dl, "Prereqs")
        dl.addStretch(1)

        # Right column: diagnostics, with the big Start/Stop toggle beneath it.
        right_col = QtWidgets.QVBoxLayout(); right_col.setSpacing(16)
        right_col.addWidget(diag, 1)
        self.power_btn = QtWidgets.QPushButton("Start"); self.power_btn.setObjectName("primary")
        self.power_btn.setProperty("power", "true")
        self.power_btn.setIcon(QtGui.QIcon(_pixmap("play", "white", 24)))
        self.power_btn.setIconSize(QtCore.QSize(22, 22))
        self.power_btn.setMinimumHeight(56)
        self.power_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.power_btn.setToolTip("Start / Stop the demo server (Ctrl+S / Ctrl+K)")
        self.power_btn.clicked.connect(self._on_power_clicked)
        right_col.addWidget(self.power_btn)

        right_box = QtWidgets.QWidget(); right_box.setLayout(right_col)
        midrow.addWidget(right_box, 2)
        v.addLayout(midrow, 1)
        return w

    def _build_access_banner(self):
        """Prominent card showing the address to type into the tablet browser."""
        banner = QtWidgets.QFrame(); banner.setObjectName("access"); _shadow(banner, alpha=40)
        h = QtWidgets.QHBoxLayout(banner); h.setContentsMargins(22, 16, 18, 16); h.setSpacing(14)

        ic = QtWidgets.QLabel(); ic.setPixmap(_pixmap("link", "white", 46, bg=AZURE))
        ic.setFixedSize(46, 46)

        tv = QtWidgets.QVBoxLayout(); tv.setSpacing(2)
        cap = QtWidgets.QLabel("TABLET ACCESS"); cap.setObjectName("accessCap")
        self.tablet_big = QtWidgets.QLabel("—"); self.tablet_big.setObjectName("accessUrl")
        self.tablet_big.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.access_hint = QtWidgets.QLabel(""); self.access_hint.setObjectName("accessHint")
        tv.addWidget(cap); tv.addWidget(self.tablet_big); tv.addWidget(self.access_hint)

        # Scan-to-open QR (white tile so it stays scannable on the dark banner);
        # click it for a large pop-up version.
        self.banner_qr = QtWidgets.QLabel(); self.banner_qr.setObjectName("qrTile")
        self.banner_qr.setFixedSize(84, 84); self.banner_qr.setAlignment(QtCore.Qt.AlignCenter)
        self._make_qr_clickable(self.banner_qr)
        scan = QtWidgets.QLabel("SCAN · CLICK TO ENLARGE"); scan.setObjectName("accessCap")
        scan.setAlignment(QtCore.Qt.AlignCenter)
        qv = QtWidgets.QVBoxLayout(); qv.setSpacing(3)
        qv.addWidget(self.banner_qr); qv.addWidget(scan)

        copy = QtWidgets.QPushButton("Copy"); copy.setObjectName("accessCopy")
        copy.clicked.connect(self._copy_url)
        # Reserve width for the wider "Copied ✓" flash so the button never resizes.
        copy.ensurePolished()
        copy.setFixedWidth(copy.fontMetrics().horizontalAdvance("Copied ✓") + 42)

        h.addWidget(ic, 0, QtCore.Qt.AlignVCenter); h.addSpacing(4)
        h.addLayout(tv); h.addStretch(1)
        h.addLayout(qv); h.addSpacing(8)
        h.addWidget(copy, 0, QtCore.Qt.AlignVCenter)
        return banner

    def _stat_card_add(self, row, icon, caption, color):
        card, val, sub, _ic = self._stat_card(icon, caption, color)
        row.addWidget(card, 1)                # equal stretch → equal-width cards
        return card, val, sub

    def _qr_card(self, caption, sub, qr_px=196):
        """A compact white card with a title, a QR placeholder and a caption.
        `qr_px` sizes the code — shrink it when several cards share a row on a
        small portable screen."""
        card = QtWidgets.QFrame(); card.setObjectName("stat"); _shadow(card)
        card.setFixedWidth(qr_px + 32)     # hug the QR; text wraps within
        v = QtWidgets.QVBoxLayout(card); v.setContentsMargins(16, 14, 16, 14); v.setSpacing(8)
        cap = QtWidgets.QLabel(caption); cap.setObjectName("qrCaption")
        qr = QtWidgets.QLabel(); qr.setObjectName("qrCard")
        qr.setFixedSize(qr_px, qr_px); qr.setAlignment(QtCore.Qt.AlignCenter)
        self._make_qr_clickable(qr)
        sub_lbl = QtWidgets.QLabel(sub); sub_lbl.setObjectName("qrSub"); sub_lbl.setWordWrap(True)
        sub_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        v.addWidget(cap); v.addWidget(qr, 0, QtCore.Qt.AlignHCenter); v.addWidget(sub_lbl)
        return card, qr, sub_lbl

    def _set_qr(self, label, data):
        """Render `data` into `label` as a QR, caching so we don't regenerate
        the (identical) pixmap on every refresh."""
        if getattr(label, "_qr_data", None) == data:
            return
        pm = _qr_pixmap(data, min(label.width(), label.height()) - 12)
        if pm is not None:
            label.setPixmap(pm)
            label._qr_data = data
        else:                       # segno missing — show a hint once
            label.setText("QR needs\n‘segno’")
            label.setWordWrap(True)

    def _make_qr_clickable(self, label):
        """Let a QR label be clicked to open a large pop-up of the same code."""
        label.setCursor(QtCore.Qt.PointingHandCursor)
        label.setToolTip("Click to enlarge")
        label.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            if getattr(obj, "_qr_data", None):
                self._show_qr_popup(obj._qr_data)
                return True
            if obj is getattr(self, "alert", None):
                self.alert.setVisible(False)
                return True
        return super().eventFilter(obj, event)

    def _show_qr_popup(self, url):
        """Modal card with a large, easy-to-scan QR for `url`."""
        title = "Trust a new tablet" if "/trust" in url else "Scan to open the demo"
        dlg = QtWidgets.QDialog(self); dlg.setObjectName("root")
        dlg.setWindowTitle(title)
        v = QtWidgets.QVBoxLayout(dlg); v.setContentsMargins(26, 24, 26, 20); v.setSpacing(14)
        cap = QtWidgets.QLabel(title); cap.setObjectName("qrCaption")
        cap.setAlignment(QtCore.Qt.AlignCenter)
        qr = QtWidgets.QLabel(); qr.setObjectName("qrCard")
        qr.setFixedSize(400, 400); qr.setAlignment(QtCore.Qt.AlignCenter)
        pm = _qr_pixmap(url, 380)
        if pm is not None:
            qr.setPixmap(pm)
        u = QtWidgets.QLabel(url); u.setObjectName("qrSub"); u.setAlignment(QtCore.Qt.AlignCenter)
        u.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        close = QtWidgets.QPushButton("Close"); close.setObjectName("primary")
        close.clicked.connect(dlg.accept)
        row = QtWidgets.QHBoxLayout(); row.addStretch(1); row.addWidget(close); row.addStretch(1)
        v.addWidget(cap); v.addWidget(qr, 0, QtCore.Qt.AlignHCenter); v.addWidget(u); v.addLayout(row)
        dlg.exec()

    # ---- demos card ------------------------------------------------------
    CHIP = {  # status -> (label, background, text colour)
        "ready":    ("READY",   "#e6f7f0", GREEN),
        "stale":    ("REBUILD", "#fff2df", AMBER),
        "build":    ("BUILD",   "#fdecea", RED),
        "degraded": ("CHECK",   "#fff2df", AMBER),
        "missing":  ("MISSING", "#fdecea", RED),
    }

    def _build_demos_box(self):
        box = QtWidgets.QGroupBox("Demos"); _shadow(box)
        v = QtWidgets.QVBoxLayout(box); v.setSpacing(10)
        self.demo_rows = []   # [(chip_label, detail_label)] in demo_status() order
        try:
            demos = doctor.demo_status()
        except Exception:
            demos = []
        for d in demos:
            item, chip, detail = self._demo_item(d["name"])
            v.addWidget(item)
            self.demo_rows.append((chip, detail))
        v.addStretch(1)
        self.rebuild_btn = QtWidgets.QPushButton("Rebuild frontend")
        self.rebuild_btn.setToolTip("Run `vite build --base /ssm/` — needed after "
                                    "the SSM UI shows REBUILD (Ctrl+Shift+R)")
        self.rebuild_btn.clicked.connect(self.rebuild_frontend)
        v.addWidget(self.rebuild_btn)
        return box

    def _demo_item(self, name):
        item = QtWidgets.QFrame(); item.setObjectName("demoItem")
        v = QtWidgets.QVBoxLayout(item); v.setContentsMargins(14, 10, 14, 10); v.setSpacing(3)
        top = QtWidgets.QHBoxLayout()
        n = QtWidgets.QLabel(name); n.setObjectName("demoName")
        chip = QtWidgets.QLabel("…"); chip.setObjectName("demoChip")
        chip.setAlignment(QtCore.Qt.AlignCenter)
        top.addWidget(n); top.addStretch(1); top.addWidget(chip)
        detail = QtWidgets.QLabel("…"); detail.setObjectName("demoDetail"); detail.setWordWrap(True)
        v.addLayout(top); v.addWidget(detail)
        return item, chip, detail

    def _apply_demos(self, demos):
        for (chip, detail), d in zip(self.demo_rows, demos):
            label, bg, fg = self.CHIP.get(d["status"], ("?", "#eef0f3", GREY))
            chip.setText(label)
            chip.setStyleSheet(
                f"background:{bg};color:{fg};font-size:10px;font-weight:800;"
                f"border-radius:10px;padding:3px 10px")
            detail.setText(f"{d['route']} · {d['detail']}")

    def _page_console(self):
        w = QtWidgets.QGroupBox("Server console"); _shadow(w)
        v = QtWidgets.QVBoxLayout(w)
        self.log = QtWidgets.QPlainTextEdit(); self.log.setObjectName("console")
        self.log.setReadOnly(True); self.log.setMaximumBlockCount(4000)
        v.addWidget(self.log)
        return w

    def _page_access(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(16)
        acc = QtWidgets.QGroupBox("Access URLs"); _shadow(acc)
        al = QtWidgets.QVBoxLayout(acc)
        self.url_lbl = QtWidgets.QLabel(""); self.url_lbl.setObjectName("url")
        self.url_lbl.setWordWrap(True)
        self.url_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        al.addWidget(self.url_lbl)
        row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open in browser"); self.open_btn.setObjectName("primary")
        self.open_btn.clicked.connect(self._open_browser)
        self.copy_btn = QtWidgets.QPushButton("Copy tablet URL")
        self.copy_btn.clicked.connect(self._copy_url)
        row.addWidget(self.open_btn); row.addWidget(self.copy_btn); row.addStretch(1)
        al.addLayout(row)
        v.addWidget(acc)

        # Scan-to-open (compact QR cards) beside the protocol cheat-sheet, so
        # the row's width is used instead of leaving the QRs floating in space.
        row2 = QtWidgets.QHBoxLayout(); row2.setSpacing(16)

        scanbox = QtWidgets.QGroupBox("Scan to open"); _shadow(scanbox)
        sl = QtWidgets.QHBoxLayout(scanbox); sl.setSpacing(16)
        # Three codes share this row, so use a slightly smaller QR to stay on a 13"
        # screen. Two demo codes: the named mmg-demo.local (iPhone/iPad resolve it
        # via Bonjour) and a direct-IP fallback for tablets — mainly Android — that
        # can't resolve .local names.
        demo_card, self.qr_demo, self.qr_demo_sub = self._qr_card(
            "Open the demo (name)", "Point a tablet camera here to open the demo.",
            qr_px=158)
        demo_card.layout().addStretch(1)     # top-align content so QRs line up
        sl.addWidget(demo_card)

        demo_ip_card, self.qr_demo_ip, self.qr_demo_ip_sub = self._qr_card(
            "Android fallback (IP)",
            "Use this if the tablet can't open the name above — mainly Android.",
            qr_px=158)
        demo_ip_card.layout().addStretch(1)
        sl.addWidget(demo_ip_card)

        trust_card, self.qr_trust, self.qr_trust_sub = self._qr_card(
            "Trust a new tablet",
            "Do this once per tablet so HTTPS shows no warning (needed for the "
            "EMG mic).", qr_px=158)
        tb = QtWidgets.QVBoxLayout(); tb.setSpacing(8)   # stack buttons in the narrow card
        self.trust_open_btn = QtWidgets.QPushButton("Open /trust"); self.trust_open_btn.setObjectName("primary")
        self.trust_open_btn.clicked.connect(self._open_trust)
        self.trust_copy_btn = QtWidgets.QPushButton("Copy /trust URL")
        self.trust_copy_btn.clicked.connect(self._copy_trust)
        tb.addWidget(self.trust_open_btn); tb.addWidget(self.trust_copy_btn)
        trust_card.layout().addLayout(tb)
        trust_card.layout().addStretch(1)    # top-align content so QRs line up
        sl.addWidget(trust_card)
        row2.addWidget(scanbox, 0)                       # hug the two compact cards

        help_box = QtWidgets.QGroupBox("Protocol cheat-sheet"); _shadow(help_box)
        hl = QtWidgets.QVBoxLayout(help_box)
        tip = QtWidgets.QLabel(
            f"{_dot(GREEN)} <b>https://&lt;host&gt;:{doctor.HTTPS_PORT}</b> — the app<br>"
            f"{_dot(GREEN)} <b>http://&lt;host&gt;:{doctor.HTTP_PORT}</b> — redirects to HTTPS<br>"
            f"{_dot(RED)} http://…:{doctor.HTTPS_PORT} or https://…:{doctor.HTTP_PORT} "
            f"— connection reset")
        tip.setObjectName("url"); tip.setTextFormat(QtCore.Qt.RichText); tip.setWordWrap(True)
        tip.setAlignment(QtCore.Qt.AlignTop)
        hl.addWidget(tip); hl.addStretch(1)
        row2.addWidget(help_box, 1)                      # fill the remaining width

        v.addLayout(row2)
        v.addStretch(1)
        return w

    def _page_doctor(self):
        w = QtWidgets.QGroupBox("Diagnostic report"); _shadow(w)
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.doctor_btn = QtWidgets.QPushButton("Run doctor"); self.doctor_btn.setObjectName("primary")
        self.doctor_btn.setToolTip("Run full diagnostics (Ctrl+D)")
        self.doctor_btn.clicked.connect(self.run_doctor)
        top.addWidget(self.doctor_btn); top.addStretch(1)
        v.addLayout(top)
        self.docout = QtWidgets.QPlainTextEdit(); self.docout.setObjectName("docout")
        self.docout.setReadOnly(True)
        self.docout.setPlaceholderText("Click “Run doctor” to run a full diagnosis…")
        v.addWidget(self.docout)
        return w

    def _control_bar(self):
        ctl = QtWidgets.QHBoxLayout(); ctl.setSpacing(10)
        # Start/Stop live in the big power toggle (Diagnostics column). The bar
        # keeps the secondary controls.
        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.reset_btn = QtWidgets.QPushButton("Reset ports"); self.reset_btn.setObjectName("danger")
        self.restart_btn.clicked.connect(self.restart_server)
        self.reset_btn.clicked.connect(self.reset_ports)
        self.restart_btn.setToolTip("Restart the server (Ctrl+R)")
        self.reset_btn.setToolTip("Force-free ports 8443 + 8000 if a server is stuck")
        for b in (self.restart_btn, self.reset_btn):
            b.setProperty("ctl", "true")      # enlarged via QSS [ctl="true"]
            b.setMinimumHeight(54)            # uniform, easy-to-hit height
            b.setCursor(QtCore.Qt.PointingHandCursor)
            ctl.addWidget(b)

        # Busy indicator — hidden until Start/Stop is pressed, shown while the
        # server subprocess is booting (until HTTPS answers 200) or shutting
        # down (until the ports go free).
        self.busy_lbl = QtWidgets.QLabel("")
        self.busy_lbl.setObjectName("startupLbl")
        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setObjectName("startup")
        self.busy_bar.setRange(0, 0)          # indeterminate / busy
        self.busy_bar.setTextVisible(False)
        self.busy_bar.setFixedWidth(150)
        self.busy_lbl.hide(); self.busy_bar.hide()
        ctl.addSpacing(6)
        ctl.addWidget(self.busy_lbl); ctl.addWidget(self.busy_bar)

        ctl.addStretch(1)
        self.keepalive = QtWidgets.QCheckBox("Keep alive")
        self.keepalive.setToolTip("Automatically restart the server if it exits unexpectedly")
        self.keepalive.toggled.connect(self._toggle_keepalive)
        ctl.addWidget(self.keepalive)
        self.auto = QtWidgets.QCheckBox("Auto-refresh every"); self.auto.setChecked(True)
        self.auto.toggled.connect(self._toggle_auto)
        self.interval = QtWidgets.QSpinBox()
        self.interval.setRange(1, 60); self.interval.setValue(3); self.interval.setSuffix(" s")
        self.interval.valueChanged.connect(lambda _: self._toggle_auto(self.auto.isChecked()))
        ctl.addWidget(self.auto); ctl.addWidget(self.interval)
        return ctl

    def _diag_row(self, lay, name):
        row = QtWidgets.QHBoxLayout()
        n = QtWidgets.QLabel(name); n.setFixedWidth(84); n.setAlignment(QtCore.Qt.AlignTop)
        n.setStyleSheet(f"color:{LABEL};font-weight:800;font-size:14px")
        val = QtWidgets.QLabel("…"); val.setObjectName("diag")
        val.setTextFormat(QtCore.Qt.RichText); val.setWordWrap(True)
        row.addWidget(n); row.addWidget(val, 1)
        lay.addLayout(row)
        return val

    # ---- logging ---------------------------------------------------------
    def _append_log(self, text):
        self.log.appendPlainText(text.rstrip())

    def _pump(self, proc):
        for ln in proc.stdout:
            self._log_sig.emit(ln)
        # stdout closed → the process has exited; report it so we can flag a
        # crash if we didn't ask it to stop.
        self._exit_sig.emit(proc, proc.wait())

    def _on_proc_exit(self, proc, rc):
        # Immediate path for a server WE started: the stdout pump saw it exit.
        # Ignore deliberate stops (marked) or an already-replaced process.
        if getattr(proc, "_intentional", False) or proc is not self.proc:
            return
        if self._starting:
            return            # startup failure is handled in _apply_status
        self.proc = None
        self._last_rc = rc
        self._maybe_crash()   # the polling path also funnels here (guarded)

    def _maybe_crash(self):
        """Handle an unexpected server exit exactly once — from either the
        stdout pump (self-started) or status polling (adopted). Fires the
        alert/beep and, if keep-alive is on, schedules a backoff restart."""
        if self._exit_handled:
            return
        self._exit_handled = True
        rc = self._last_rc
        code = f" (exit code {rc})" if rc is not None else ""
        self._append_log(f"[dashboard] ⚠ server exited unexpectedly{code}")
        QtWidgets.QApplication.beep()
        if self._keepalive and self._crash_count < self.MAX_RESTARTS:
            self._crash_count += 1
            delay = min(2 ** (self._crash_count - 1), 30)   # 1,2,4,8,16,30s…
            self._append_log(f"[dashboard] keep-alive: restarting in {delay}s "
                             f"(attempt {self._crash_count}/{self.MAX_RESTARTS})")
            self._show_alert(f"Server exited{code}. Auto-restarting in {delay}s "
                             f"(attempt {self._crash_count}/{self.MAX_RESTARTS})…")
            QtCore.QTimer.singleShot(int(delay * 1000), self._auto_restart)
        elif self._keepalive:
            self._show_alert(f"Server keeps exiting — keep-alive gave up after "
                             f"{self.MAX_RESTARTS} attempts. Check the Console.")
            self._append_log("[dashboard] keep-alive: gave up after "
                             f"{self.MAX_RESTARTS} attempts")
        else:
            self._show_alert(f"Server exited unexpectedly{code}. "
                             f"See the Console for details.")
        self.refresh()

    def _auto_restart(self):
        if not self._keepalive:
            return                            # toggled off during the backoff wait
        if self.proc and self.proc.poll() is None:
            return                            # already back up
        if doctor.find_listeners([doctor.HTTPS_PORT]).get(doctor.HTTPS_PORT):
            return                            # something already on the port
        self._append_log("[dashboard] keep-alive: restarting server…")
        self.start_server()

    def _show_alert(self, msg):
        self.alert.setText(f"⚠  {msg}   (click to dismiss)")
        self.alert.setVisible(True)

    # ---- status value that auto-fits the card width ---------------------
    MAX_STATUS_PX, MIN_STATUS_PX = 24, 13

    def _set_status_text(self, text, col):
        self._status_text = text
        self._status_col = col
        self.k_status.setText(text)
        self._refit_status()

    def _refit_status(self):
        """Shrink the STATUS word until it fits the card (measured against the
        *card* width, not the label's own — the label hugs its text)."""
        if not hasattr(self, "_status_text"):
            return
        # card width minus icon (44), its spacing (12) and margins (28) + slack
        avail = self.c_status.width() - 88
        px = self.MAX_STATUS_PX
        if avail >= 24:
            f = QtGui.QFont(self.k_status.font()); f.setBold(True)
            px = self.MIN_STATUS_PX
            for size in range(self.MAX_STATUS_PX, self.MIN_STATUS_PX - 1, -1):
                f.setPixelSize(size)
                if QtGui.QFontMetrics(f).horizontalAdvance(self._status_text) <= avail:
                    px = size
                    break
        self.k_status.setStyleSheet(
            f"font-size:{px}px;font-weight:800;color:{self._status_col}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refit_status()               # keep the status word fitting the card

    # ---- status refresh --------------------------------------------------
    def refresh(self):
        if self._polling:
            return
        self._polling = True
        self.worker = StatusWorker()
        self.worker.done.connect(self._apply_status)
        self.worker.start()

    def _apply_status(self, s):
        self._polling = False
        running = s["running"]

        # --- monitor any server, even one this instance didn't start ---------
        if running and not self._was_running:
            # a fresh run appeared → (re)arm crash detection
            self._exit_handled = False
            self._last_rc = None
        if running and self.proc is None and not self._starting:
            pid = s["pids"][0] if s.get("pids") else None
            if pid and pid != self._adopted_pid:
                self._adopted_pid = pid
                self._append_log(f"[dashboard] found a running server (PID {pid}) — "
                                 f"monitoring it; Stop / Restart will control it")
        if not running:
            self._adopted_pid = None
        # Unexpected exit (covers adopted servers, and self-started ones if the
        # pump is slow): running → stopped without us asking. Intentional stops
        # are excluded by the transition guards below.
        if (self._was_running and not running and not self._starting
                and not self._stopping and not self._restart_pending):
            self._maybe_crash()
        self._was_running = running

        col = GREEN if running else GREY

        self.chip.setText("● RUNNING" if running else "○ STOPPED")
        self.chip.setStyleSheet(
            f"background:{'#e6f7f0' if running else '#eef0f3'};color:{col};"
            f"font-size:12px;font-weight:700;border-radius:13px;padding:5px 14px")

        self._set_status_text("RUNNING" if running else "STOPPED", col)
        self.status_icon.setPixmap(_pixmap("status", "white", 44, bg=col))

        if running and s["create_time"]:
            secs = int(time.time() - s["create_time"])
            self.k_uptime.setText(f"{secs//3600}h {secs%3600//60}m" if secs >= 3600
                                  else f"{secs//60}m {secs%60}s")
        else:
            self.k_uptime.setText("—")

        self.k_clients.setText(str(s["clients"]) if s["clients"] is not None else "—")

        cpu = s.get("cpu")
        self.k_cpu.setText(f"{cpu:.0f}%" if cpu is not None else "—")
        srv_cpu = s.get("srv_cpu")
        self.k_cpu_sub.setText(f"server {srv_cpu:.0f}%" if srv_cpu is not None
                               else ("server —" if running else "server off"))

        gpu = s.get("gpu")
        if gpu is not None:
            self.k_gpu.setText(f"{gpu:.0f}%")
            self.k_gpu.setStyleSheet("")
        else:
            self.k_gpu.setText("N/A")
            self.k_gpu.setStyleSheet(f"font-size:24px;font-weight:800;color:{GREY}")
        srv_gpu = s.get("srv_gpu")
        self.k_gpu_sub.setText(f"server {srv_gpu:.0f}%" if srv_gpu is not None
                               else ("server —" if running else "server off"))

        self._apply_demos(s.get("demos", []))

        if running:
            self.ports_lbl.setText(
                f"{_dot(GREEN if s['port_https'] else RED)} 8443 &nbsp; "
                f"{_dot(GREEN if s['port_http'] else RED)} 8000")
            hok = s["https_code"] == 200
            rok = s["http_code"] in (301, 302)
            self.health_lbl.setText(
                f"{_dot(GREEN if hok else RED)} https {s['https_code']} &nbsp; "
                f"{_dot(GREEN if rok else RED)} http {s['http_code']}")
        else:
            self.ports_lbl.setText(f'<span style="color:{GREY}">8443 / 8000 free</span>')
            self.health_lbl.setText(f'<span style="color:{GREY}">—</span>')

        if not running:
            self.mdns_lbl.setText(f'<span style="color:{GREY}">—</span>')
        elif s.get("mdns_ok"):
            self.mdns_lbl.setText(f"{_dot(GREEN)} mmg-demo.local resolves")
        else:
            self.mdns_lbl.setText(
                f"{_dot(AMBER)} mmg-demo.local not resolving — tablets use the IP")

        self.deps_lbl.setText(
            f"{_dot(GREEN if s['dist'] else RED)} dist &nbsp; "
            f"{_dot(GREEN if s['model'] else RED)} model &nbsp; "
            f"{_dot(GREEN if s['bones'] else AMBER)} bones &nbsp; "
            f"{_dot(GREEN if s['cert'] else AMBER)} cert")

        ip = s["ip"]
        self._url = f"https://localhost:{doctor.HTTPS_PORT}"
        self._tablet_url = f"https://{doctor.MDNS_FQDN}:{doctor.HTTPS_PORT}"
        self._tablet_ip_url = f"https://{ip}:{doctor.HTTPS_PORT}"
        self.tablet_big.setText(self._tablet_url)
        self.access_hint.setText(
            f"Open in the tablet browser · use {self._tablet_ip_url} if the name won't resolve"
            if running else "Start the server, then open this on the tablet")
        self.url_lbl.setText(
            f"This device &nbsp;<b>{self._url}</b><br>"
            f"Tablets &nbsp;&nbsp;&nbsp;&nbsp;<b>{self._tablet_url}</b> "
            f"&nbsp;(or <b>{self._tablet_ip_url}</b> — by IP; "
            f"http://{ip}:{doctor.HTTP_PORT} redirects)")

        # Two demo QRs: the named .local URL (iPhone/iPad resolve it via Bonjour)
        # and a direct-IP fallback for tablets — mainly Android — that can't
        # resolve .local. The small banner QR stays on the IP, since it's the one
        # code that works on any device. Updated only when the underlying URL
        # changes (see _set_qr caching).
        self._trust_url = f"http://{ip}:{doctor.HTTP_PORT}/trust"
        self._set_qr(self.banner_qr, self._tablet_ip_url)
        self._set_qr(self.qr_demo, self._tablet_url)
        self._set_qr(self.qr_demo_ip, self._tablet_ip_url)
        self._set_qr(self.qr_trust, self._trust_url)
        self.qr_demo_sub.setText(
            f"Point a tablet camera here to open the demo.\n{self._tablet_url}")
        self.qr_demo_ip_sub.setText(
            f"Use this if the name won't open — mainly Android.\n{self._tablet_ip_url}")
        self.qr_trust_sub.setText(
            f"Install the cert once per tablet (needed for the EMG mic).\n{self._trust_url}")

        self._update_power(running)

        if self._starting:
            ready = running and s.get("https_code") == 200
            died = self.proc is not None and self.proc.poll() is not None
            timed_out = time.time() - self._start_t0 > 45
            if ready or died or timed_out:
                self._finish_startup(ready)
            else:
                # Keep the "starting" affordances until the server truly answers.
                self.chip.setText("◌ STARTING")
                self.chip.setStyleSheet(
                    f"background:#fff2df;color:{AMBER};font-size:12px;"
                    f"font-weight:700;border-radius:13px;padding:5px 14px")
                self._set_status_text("STARTING", AMBER)
                self.status_icon.setPixmap(_pixmap("status", "white", 44, bg=AMBER))
                self.power_btn.setText("Starting…"); self.power_btn.setEnabled(False)

        if self._stopping:
            elapsed = time.time() - self._stop_t0
            if not running:
                self._finish_stopping(True)
            elif elapsed > 8:
                self._finish_stopping(False)
            else:
                # Force-kill a process that ignored terminate() before giving up.
                if elapsed > 4 and self.proc is not None and self.proc.poll() is None:
                    self.proc.kill()
                    self.proc = None
                self.chip.setText("◌ STOPPING")
                self.chip.setStyleSheet(
                    f"background:#fff2df;color:{AMBER};font-size:12px;"
                    f"font-weight:700;border-radius:13px;padding:5px 14px")
                self._set_status_text("STOPPING", AMBER)
                self.status_icon.setPixmap(_pixmap("status", "white", 44, bg=AMBER))
                self.power_btn.setText("Stopping…"); self.power_btn.setEnabled(False)

        # keep-alive: a run that stays healthy long enough clears the backoff.
        if (running and s.get("https_code") == 200 and self._crash_count
                and time.time() - self._last_restart_t > 45):
            self._crash_count = 0

        # autostart: launch once on the first status, only if nothing is up yet.
        if self._autostart and not self._autostart_done and not self._starting:
            self._autostart_done = True
            if not running:
                self._append_log("[dashboard] autostart: launching server…")
                self._on_start_clicked()

    # ---- controls --------------------------------------------------------
    def start_server(self):
        if doctor.find_listeners([doctor.HTTPS_PORT]).get(doctor.HTTPS_PORT):
            # Port is busy — but is it *our* server, or something else squatting?
            if doctor.find_server_listeners([doctor.HTTPS_PORT]).get(doctor.HTTPS_PORT):
                self._append_log("[dashboard] already running — skipping start")
            else:
                self._append_log("[dashboard] port 8443 is held by another process "
                                 "(not the MMG server) — use Reset ports to free it")
            self.refresh(); return
        self._append_log("[dashboard] starting server…")
        self.alert.setVisible(False)          # clear any prior crash notice
        self._last_restart_t = time.time()    # baseline for keep-alive backoff reset
        self.proc = subprocess.Popen(
            [sys.executable, doctor.SERVER_PY],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=doctor.HERE)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self._begin_startup()

    def _begin_startup(self):
        """Show the busy indicator and poll (off-thread) until the server is
        actually serving, rather than assuming a fixed delay."""
        self._stopping = False
        self._starting = True
        self._start_t0 = time.time()
        self._begin_busy("Starting server…")

    def _finish_startup(self, ok):
        self._starting = False
        self._end_busy()
        self._append_log("[dashboard] server ready"
                         if ok else "[dashboard] startup did not complete — check console")

    def stop_server(self):
        self._append_log("[dashboard] stopping server…")
        # Ask the process to exit (non-blocking) — teardown is then tracked by
        # polling in _apply_status so the GUI never freezes.
        if self.proc and self.proc.poll() is None:
            self.proc._intentional = True     # so its exit isn't flagged as a crash
            self.proc.terminate()
        else:
            self._kill_listeners()
        self._begin_teardown("Stopping server…",
                             "[dashboard] server stopped",
                             "[dashboard] stop timed out — try Reset ports")

    def _begin_teardown(self, message, ok_msg, fail_msg):
        """Enter a 'waiting for the ports to go free' state with the busy
        indicator, used by both Stop and Reset ports."""
        self._starting = False
        self._stopping = True
        self._stop_t0 = time.time()
        self._stop_ok_msg = ok_msg
        self._stop_fail_msg = fail_msg
        self._begin_busy(message)

    def _finish_stopping(self, ok):
        self._stopping = False
        self._append_log(self._stop_ok_msg if ok else self._stop_fail_msg)
        if self._restart_pending:
            self._restart_pending = False
            if ok:
                self.start_server()   # ports are free — boot straight away
                return
        self._end_busy()

    def _kill_listeners(self, only_ours=True):
        # only_ours=True (Stop): touch just our own server.py. False (Reset ports):
        # free the ports outright, whatever process is squatting on them.
        if self.proc is not None:
            self.proc._intentional = True     # its exit is deliberate, not a crash
        find = doctor.find_server_listeners if only_ours else doctor.find_listeners
        listeners = find([doctor.HTTPS_PORT, doctor.HTTP_PORT])
        pids = {pid for hs in listeners.values() for pid, _, _ in hs}
        for pid in sorted(pids):
            ok = doctor.kill_pid(pid)
            self._append_log(f"[dashboard] stop PID {pid}: {'ok' if ok else 'FAILED'}")
        self.proc = None

    # ---- shared busy indicator ------------------------------------------
    def _begin_busy(self, message):
        self.busy_lbl.setText(message)
        self.busy_lbl.show(); self.busy_bar.show()
        for b in (self.power_btn, self.restart_btn):
            b.setEnabled(False)
        if not hasattr(self, "_busy_timer"):
            self._busy_timer = QtCore.QTimer(self)
            self._busy_timer.timeout.connect(self.refresh)
        self._busy_timer.start(600)
        self.refresh()

    def _end_busy(self):
        if hasattr(self, "_busy_timer"):
            self._busy_timer.stop()
        self.busy_lbl.hide(); self.busy_bar.hide()
        self.restart_btn.setEnabled(True)

    def restart_server(self):
        self._crash_count = 0                 # manual action = fresh slate
        running = bool(self.proc and self.proc.poll() is None) or \
            bool(any(doctor.find_server_listeners(
                [doctor.HTTPS_PORT, doctor.HTTP_PORT]).values()))
        if not running:
            self.start_server(); return
        self._append_log("[dashboard] restarting server…")
        # Stop first; once the ports go free, _finish_stopping boots it again —
        # keyed on real state, not a fixed delay.
        self._restart_pending = True
        self.stop_server()

    def reset_ports(self, silent=False):
        # silent=True is the internal path (no listeners tracked); the button
        # path shows the busy indicator and waits for the ports to go free.
        if silent:
            self._kill_listeners(only_ours=False)
            QtCore.QTimer.singleShot(800, self.refresh)
            return
        self._append_log("[dashboard] freeing ports 8443 + 8000…")
        self._kill_listeners(only_ours=False)
        self._begin_teardown("Resetting ports…",
                             "[dashboard] ports freed",
                             "[dashboard] some ports still busy — check console")

    def run_doctor(self):
        self.doctor_btn.setEnabled(False)
        self.docout.setPlainText("Running doctor…\n")
        self._dw = DoctorWorker()
        self._dw.done.connect(self._doctor_done)
        self._dw.start()

    def _doctor_done(self, text):
        self.docout.setPlainText(text)
        self.doctor_btn.setEnabled(True)

    def rebuild_frontend(self):
        if self._rebuilding:
            return
        if not os.path.isdir(doctor.GUI_DIR):
            self._append_log("[dashboard] TauriGUI dir not found — cannot rebuild")
            return
        self._rebuilding = True
        self.rebuild_btn.setEnabled(False); self.rebuild_btn.setText("Rebuilding…")
        self._go(1)                          # show the Console so output is visible
        self._append_log("[dashboard] rebuilding SSM frontend (vite build --base /ssm/)…")
        self._rw = RebuildWorker(doctor.GUI_DIR)
        self._rw.line.connect(self._append_log)
        self._rw.done.connect(self._rebuild_done)
        self._rw.start()

    def _rebuild_done(self, code):
        self._rebuilding = False
        self.rebuild_btn.setEnabled(True); self.rebuild_btn.setText("Rebuild frontend")
        self._append_log(f"[dashboard] rebuild {'complete' if code == 0 else f'failed (code {code})'}")
        self.refresh()                       # refresh the demo build-status chips

    # ---- power toggle (Start / Stop in one button) ----------------------
    def _running_now(self):
        return bool(self.proc and self.proc.poll() is None) or \
            bool(doctor.find_listeners([doctor.HTTPS_PORT]).get(doctor.HTTPS_PORT))

    def _on_power_clicked(self):
        if self._running_now():
            self.stop_server()
        else:
            self._on_start_clicked()

    def _update_power(self, running):
        """Point the single Start/Stop button at the right action + look."""
        want = "lava" if running else "primary"
        if self.power_btn.objectName() != want:
            self.power_btn.setObjectName(want)
            self._repolish(self.power_btn)     # re-apply QSS after id change
        self.power_btn.setText("Stop" if running else "Start")
        self.power_btn.setIcon(QtGui.QIcon(
            _pixmap("stopsq" if running else "play", "white", 24)))
        self.power_btn.setEnabled(True)

    @staticmethod
    def _repolish(w):
        w.style().unpolish(w); w.style().polish(w); w.update()

    # ---- misc ------------------------------------------------------------
    def _on_start_clicked(self):
        self._crash_count = 0                 # manual start = fresh slate
        self.start_server()

    def _toggle_keepalive(self, on):
        self._keepalive = on
        self._crash_count = 0
        self._append_log(f"[dashboard] keep-alive {'on' if on else 'off'}")

    def _toggle_auto(self, on):
        if on:
            self.timer.start(self.interval.value() * 1000)
        else:
            self.timer.stop()

    def _open_browser(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._url))

    def _copy_url(self):
        QtWidgets.QApplication.clipboard().setText(self._tablet_url)
        self._append_log("[dashboard] tablet URL copied to clipboard")
        self._flash_copied(self.sender())

    def _open_trust(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._trust_url))

    def _copy_trust(self):
        QtWidgets.QApplication.clipboard().setText(self._trust_url)
        self._append_log("[dashboard] /trust URL copied to clipboard")
        self._flash_copied(self.sender())

    def _flash_copied(self, btn):
        """Briefly change a Copy button to 'Copied ✓' so the copy is obvious."""
        if not isinstance(btn, QtWidgets.QPushButton):
            return
        if getattr(btn, "_orig_text", None) is None:
            btn._orig_text = btn.text()
        btn.setText("Copied ✓")
        QtCore.QTimer.singleShot(1500, lambda: btn.setText(btn._orig_text))

    def closeEvent(self, e):
        if self.proc and self.proc.poll() is None:
            r = QtWidgets.QMessageBox.question(
                self, "Server still running",
                "Stop the demo server before closing the panel?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel)
            if r == QtWidgets.QMessageBox.Cancel:
                e.ignore(); return
            if r == QtWidgets.QMessageBox.Yes:
                self.proc.terminate()
        e.accept()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MMG Demo Server control panel")
    ap.add_argument("--autostart", action="store_true",
                    help="start the demo server automatically once the panel opens")
    ap.add_argument("--keep-alive", action="store_true",
                    help="auto-restart the server if it exits unexpectedly")
    ap.add_argument("--fullscreen", action="store_true",
                    help="open the panel in full screen (F11 or Esc to leave)")
    args = ap.parse_args()

    # Consistent high-DPI scaling so the window doesn't jump size when it's
    # dragged across monitors with different Windows scale factors (a laptop +
    # projector at an event) or on a fractional-scaled single screen. Must be
    # set before the QApplication is created.
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QtWidgets.QApplication(sys.argv)
    try:
        from qt_material import apply_stylesheet
        apply_stylesheet(app, theme="light_blue.xml")
    except Exception:
        pass
    app.setStyleSheet(app.styleSheet() + QSS)
    win = Dashboard(autostart=args.autostart, keepalive=args.keep_alive)

    # Size to fit the actual screen (small portable laptops) and centre it,
    # so the larger type never pushes content off-screen.
    scr = app.primaryScreen().availableGeometry()
    w = min(1200, int(scr.width() * 0.94))
    h = min(760, int(scr.height() * 0.94))
    win.resize(w, h)
    win.move(scr.center().x() - w // 2, scr.center().y() - h // 2)

    if args.fullscreen:
        win.showFullScreen()
        win._sync_fs_btn()
    else:
        win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
