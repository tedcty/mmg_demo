#!/usr/bin/env python3
"""
dashboard.py — Material-styled desktop control panel for the MMG Demo Server.

A standalone operator window: start / stop / restart the server, free the
ports, and watch live status (ports, prerequisites, protocol health, URLs)
with optional auto-refresh. All the underlying logic is reused from doctor.py.

  python dashboard.py

Built on PySide6 (already in the demo env via ptb_mmg) + qt-material for the
Material Design theme. Run it in the `demo` conda env so it can launch the
server with the right interpreter:

  conda run -n demo python DemoServer/dashboard.py
"""

import os
import subprocess
import sys
import threading
import time

from PySide6 import QtCore, QtGui, QtWidgets

import doctor  # same folder — reuse all status/reset logic

# Brand-ish colours (UoA 2025): Whenua green / Lava / muted grey.
GREEN, RED, GREY, AMBER = "#12b57f", "#ff5c39", "#8a8f98", "#e0a100"


def _dot(ok):
    return f'<span style="color:{GREEN if ok else RED};font-size:15px">●</span>'


class StatusWorker(QtCore.QThread):
    """Gather server status off the UI thread (protocol probe can block)."""
    done = QtCore.Signal(dict)

    def run(self):
        ports = [doctor.HTTPS_PORT, doctor.HTTP_PORT]
        listeners = doctor.find_listeners(ports)
        pids = sorted({pid for hs in listeners.values() for pid, _, _ in hs})
        s = {
            "running": bool(pids),
            "pids": pids,
            "port_https": bool(listeners.get(doctor.HTTPS_PORT)),
            "port_http": bool(listeners.get(doctor.HTTP_PORT)),
            "dist": os.path.isdir(doctor.VITE_DIST),
            "model": os.path.isdir(doctor.SSM_MODEL) and os.path.exists(doctor.ANTHRO),
            "bones": os.path.exists(doctor.BONES),
            "cert": os.path.exists(doctor.CERT_FILE) and os.path.exists(doctor.KEY_FILE),
            "ip": doctor.lan_ip(),
            "https_code": None,
            "http_code": None,
            "create_time": None,
        }
        if s["running"]:
            s["https_code"], _ = doctor._get(f"https://localhost:{doctor.HTTPS_PORT}/", timeout=2)
            s["http_code"], _ = doctor._get(f"http://localhost:{doctor.HTTP_PORT}/", timeout=2)
            try:
                import psutil
                s["create_time"] = psutil.Process(pids[0]).create_time()
            except Exception:
                pass
        self.done.emit(s)


class DoctorWorker(QtCore.QThread):
    """Run doctor.diagnose() off the UI thread, capturing its printed report."""
    done = QtCore.Signal(str)

    def __init__(self, want_https=True):
        super().__init__()
        self.want_https = want_https

    def run(self):
        import contextlib, io
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                doctor.diagnose(self.want_https)
        except Exception as e:
            buf.write(f"\n[doctor error] {e}\n")
        self.done.emit(buf.getvalue())


class Dashboard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMG Demo Server — Control Panel")
        self.resize(560, 720)
        self.proc = None            # server subprocess we launched (if any)
        self._our_start = None      # our launch time, for uptime
        self._polling = False
        self._url = f"https://localhost:{doctor.HTTPS_PORT}"

        self._build_ui()

        self._log_sig.connect(self._append_log)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self._toggle_auto(self.auto.isChecked())
        self.refresh()

    # ---- UI construction -------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Header: status pill + PID/uptime
        self.pill = QtWidgets.QLabel("…")
        self.pill.setAlignment(QtCore.Qt.AlignCenter)
        self.pill.setFixedHeight(40)
        self.pill.setStyleSheet(self._pill_qss(GREY))
        font = self.pill.font(); font.setPointSize(13); font.setBold(True)
        self.pill.setFont(font)
        root.addWidget(self.pill)

        self.subline = QtWidgets.QLabel("")
        self.subline.setAlignment(QtCore.Qt.AlignCenter)
        self.subline.setStyleSheet(f"color:{GREY}")
        root.addWidget(self.subline)

        # Status cards
        self.ports_lbl = self._card(root, "Ports")
        self.deps_lbl = self._card(root, "Prerequisites")
        self.health_lbl = self._card(root, "Protocol health")

        # URLs card with copy/open
        url_box = QtWidgets.QGroupBox("Open the demo at")
        ul = QtWidgets.QVBoxLayout(url_box)
        self.url_lbl = QtWidgets.QLabel("")
        self.url_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.url_lbl.setWordWrap(True)
        ul.addWidget(self.url_lbl)
        row = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton("Open in browser")
        self.open_btn.clicked.connect(self._open_browser)
        self.copy_btn = QtWidgets.QPushButton("Copy tablet URL")
        self.copy_btn.clicked.connect(self._copy_url)
        row.addWidget(self.open_btn); row.addWidget(self.copy_btn)
        ul.addLayout(row)
        root.addWidget(url_box)

        # Controls
        ctl = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.reset_btn = QtWidgets.QPushButton("Reset ports")
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn.clicked.connect(self.stop_server)
        self.restart_btn.clicked.connect(self.restart_server)
        self.reset_btn.clicked.connect(self.reset_ports)
        for b in (self.start_btn, self.stop_btn, self.restart_btn, self.reset_btn):
            ctl.addWidget(b)
        root.addLayout(ctl)

        # Auto-refresh toggle + manual refresh
        ar = QtWidgets.QHBoxLayout()
        self.auto = QtWidgets.QCheckBox("Auto-refresh every")
        self.auto.setChecked(True)
        self.auto.toggled.connect(self._toggle_auto)
        self.interval = QtWidgets.QSpinBox()
        self.interval.setRange(1, 60); self.interval.setValue(3); self.interval.setSuffix(" s")
        self.interval.valueChanged.connect(lambda _: self._toggle_auto(self.auto.isChecked()))
        self.doctor_btn = QtWidgets.QPushButton("Run doctor")
        self.doctor_btn.clicked.connect(self.run_doctor)
        self.refresh_btn = QtWidgets.QPushButton("Refresh now")
        self.refresh_btn.clicked.connect(self.refresh)
        ar.addWidget(self.auto); ar.addWidget(self.interval); ar.addStretch(1)
        ar.addWidget(self.doctor_btn); ar.addWidget(self.refresh_btn)
        root.addLayout(ar)

        # Log
        root.addWidget(QtWidgets.QLabel("Server log"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setStyleSheet("font-family:monospace;font-size:11px")
        root.addWidget(self.log, 1)

    def _card(self, root, title):
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        lbl = QtWidgets.QLabel("…")
        lbl.setTextFormat(QtCore.Qt.RichText)
        lay.addWidget(lbl)
        root.addWidget(box)
        return lbl

    @staticmethod
    def _pill_qss(color):
        return (f"background:{color};color:white;border-radius:20px;"
                f"padding:6px 18px;")

    # ---- logging (thread-safe) -------------------------------------------
    _log_sig = QtCore.Signal(str)

    def _append_log(self, text):
        self.log.appendPlainText(text.rstrip())

    def _pump(self, proc):
        for ln in proc.stdout:
            self._log_sig.emit(ln)

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

        self.pill.setStyleSheet(self._pill_qss(GREEN if running else GREY))
        self.pill.setText("● RUNNING" if running else "○ STOPPED")

        if running:
            up = ""
            if s["create_time"]:
                secs = int(time.time() - s["create_time"])
                up = f" · up {secs//3600}h {secs%3600//60}m {secs%60}s" if secs >= 3600 \
                     else f" · up {secs//60}m {secs%60}s"
            self.subline.setText(f"PID {', '.join(map(str, s['pids']))}{up}")
        else:
            self.subline.setText("not running")

        self.ports_lbl.setText(
            f"{_dot(not running or s['port_https'])} 8443 (HTTPS)"
            f" &nbsp;&nbsp; {_dot(not running or s['port_http'])} 8000 (redirect)"
            if running else
            f'<span style="color:{GREY}">8443 free &nbsp;&nbsp; 8000 free</span>')

        self.deps_lbl.setText(
            f"{_dot(s['dist'])} SSM dist &nbsp; {_dot(s['model'])} model+data &nbsp; "
            f"{_dot(s['bones'])} bones.json &nbsp; {_dot(s['cert'])} TLS cert")

        if running:
            hok = s["https_code"] == 200
            rok = s["http_code"] in (301, 302)
            self.health_lbl.setText(
                f"{_dot(hok)} https→{s['https_code']} &nbsp;&nbsp; "
                f"{_dot(rok)} http→{s['http_code']} (redirect)")
        else:
            self.health_lbl.setText(f'<span style="color:{GREY}">— server not running —</span>')

        ip = s["ip"]
        self._url = f"https://localhost:{doctor.HTTPS_PORT}"
        self._tablet_url = f"https://{ip}:{doctor.HTTPS_PORT}"
        self.url_lbl.setText(
            f"This device: <b>{self._url}</b><br>"
            f"Tablets: <b>{self._tablet_url}</b> "
            f"(or http://{ip}:{doctor.HTTP_PORT} — redirects)<br>"
            f'<span style="color:{RED}">Never http://…:8443 or https://…:8000 '
            f"(connection reset)</span>")

        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.restart_btn.setEnabled(True)

    # ---- controls --------------------------------------------------------
    def start_server(self):
        if doctor.find_listeners([doctor.HTTPS_PORT]).get(doctor.HTTPS_PORT):
            self._append_log("[dashboard] already running — skipping start")
            self.refresh(); return
        self._append_log("[dashboard] starting server…")
        self.proc = subprocess.Popen(
            [sys.executable, doctor.SERVER_PY],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=doctor.HERE)
        self._our_start = time.time()
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        QtCore.QTimer.singleShot(1500, self.refresh)

    def stop_server(self):
        self._append_log("[dashboard] stopping server…")
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        else:
            self.reset_ports(silent=True)   # not ours — free by port
        QtCore.QTimer.singleShot(800, self.refresh)

    def restart_server(self):
        self.stop_server()
        QtCore.QTimer.singleShot(1500, self.start_server)

    def reset_ports(self, silent=False):
        if not silent:
            self._append_log("[dashboard] freeing ports 8443 + 8000…")
        listeners = doctor.find_listeners([doctor.HTTPS_PORT, doctor.HTTP_PORT])
        pids = {pid for hs in listeners.values() for pid, _, _ in hs}
        for pid in sorted(pids):
            ok = doctor.kill_pid(pid)
            self._append_log(f"[dashboard] stop PID {pid}: {'ok' if ok else 'FAILED'}")
        self.proc = None
        QtCore.QTimer.singleShot(800, self.refresh)

    # ---- misc ------------------------------------------------------------
    def run_doctor(self):
        self.doctor_btn.setEnabled(False)
        self._append_log("\n" + "=" * 40 + "\n[dashboard] running doctor…")
        self._dw = DoctorWorker(want_https=True)
        self._dw.done.connect(self._doctor_done)
        self._dw.start()

    def _doctor_done(self, text):
        self._append_log(text)
        self.doctor_btn.setEnabled(True)

    def _toggle_auto(self, on):
        if on:
            self.timer.start(self.interval.value() * 1000)
        else:
            self.timer.stop()

    def _open_browser(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._url))

    def _copy_url(self):
        QtWidgets.QApplication.clipboard().setText(getattr(self, "_tablet_url", self._url))
        self._append_log("[dashboard] tablet URL copied to clipboard")

    def closeEvent(self, e):
        # Leave a server we started running? Ask, so closing the panel doesn't
        # silently kill the demo mid-event.
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
    app = QtWidgets.QApplication(sys.argv)
    try:
        from qt_material import apply_stylesheet
        apply_stylesheet(app, theme="light_blue.xml")
    except Exception:
        pass   # fall back to the default style if qt-material isn't installed
    win = Dashboard()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
