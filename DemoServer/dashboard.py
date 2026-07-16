#!/usr/bin/env python3
"""
dashboard.py — Material admin-style control panel for the MMG Demo Server.

A standalone operator app laid out like a monitoring dashboard: a dark sidebar
with page navigation, KPI stat cards, a live CPU chart, diagnostics, protocol
health, an access panel and a console — with server control and optional
auto-refresh. All the underlying logic is reused from doctor.py.

  conda run -n demo python DemoServer/dashboard.py

Built on PySide6 (already in the demo env via ptb_mmg) + QtCharts + qt-material,
with a custom stylesheet layered on top. Run it in the `demo` conda env so it
can launch the server with the right interpreter.
"""

import os
import subprocess
import sys
import threading
import time

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from PySide6 import QtCharts
    HAVE_CHARTS = True
except Exception:
    HAVE_CHARTS = False

import doctor  # same folder — reuse all status/reset logic

# ---- palette (UoA 2025 brand) -------------------------------------------
AZURE  = "#1f2bd4"
INK    = "#0c0c48"
NAVY   = "#0c0c48"   # sidebar
GREEN  = "#12b57f"
RED    = "#e5484d"
AMBER  = "#e0a100"
PURPLE = "#7c4dff"
GREY   = "#8a8f98"
BG     = "#eef1f6"
CARD   = "#ffffff"
BORDER = "#e6e9f0"

QSS = f"""
QWidget#root {{ background: {BG}; }}

/* Sidebar */
QFrame#sidebar {{ background: {NAVY}; }}
QLabel#brand {{ color: white; font-size: 16px; font-weight: 800; }}
QLabel#brandSub {{ color: rgba(255,255,255,0.55); font-size: 10px; font-weight: 600; }}
QPushButton#nav {{ color: #c7cbe8; background: transparent; border: none;
                   text-align: left; padding: 11px 16px; border-radius: 10px;
                   font-size: 13px; font-weight: 600; }}
QPushButton#nav:hover {{ background: rgba(255,255,255,0.07); color: white; }}
QPushButton#nav:checked {{ background: {AZURE}; color: white; }}
QLabel#sideFoot {{ color: rgba(255,255,255,0.4); font-size: 10px; }}

/* Header */
QLabel#h1 {{ color: {INK}; font-size: 20px; font-weight: 800; }}
QLabel#chip {{ font-size: 12px; font-weight: 700; border-radius: 13px; padding: 5px 14px; }}

/* Cards */
QFrame#stat, QGroupBox {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#statValue {{ font-size: 24px; font-weight: 800; color: {INK}; }}
QLabel#statCaption {{ color: {GREY}; font-size: 10px; font-weight: 700; }}
QGroupBox {{ margin-top: 14px; padding: 14px; font-weight: 700; color: {INK}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 2px 6px; color: {GREY}; }}
QLabel#diag {{ font-size: 13px; color: {INK}; }}
QLabel#url {{ font-size: 12px; color: {INK}; }}

/* Buttons */
QPushButton {{ background: #f1f4f9; color: {INK}; border: 1px solid #dde3ec;
               border-radius: 10px; padding: 9px 16px; font-weight: 700; }}
QPushButton:hover {{ background: #e7ecf5; }}
QPushButton:disabled {{ color: #b3bac6; background: #f4f6fa; }}
QPushButton#primary {{ background: {AZURE}; color: white; border: none; }}
QPushButton#primary:hover {{ background: #1a24b0; }}
QPushButton#primary:disabled {{ background: #b9bdec; }}
QPushButton#danger {{ background: #fdecea; color: {RED}; border: 1px solid #f6c9c7; }}
QPushButton#danger:hover {{ background: #fadedb; }}

QPlainTextEdit#console, QPlainTextEdit#docout {{
    background: #0f1424; color: #c7d2e0; border: none; border-radius: 12px;
    padding: 8px; font-family: monospace; font-size: 11px; }}
QCheckBox {{ color: {INK}; font-weight: 600; }}
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


def _shadow(w, blur=24, alpha=26, dy=4):
    eff = QtWidgets.QGraphicsDropShadowEffect(w)
    eff.setBlurRadius(blur); eff.setXOffset(0); eff.setYOffset(dy)
    eff.setColor(QtGui.QColor(12, 12, 72, alpha))
    w.setGraphicsEffect(eff)


class StatusWorker(QtCore.QThread):
    done = QtCore.Signal(dict)

    def run(self):
        ports = [doctor.HTTPS_PORT, doctor.HTTP_PORT]
        listeners = doctor.find_listeners(ports)
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
            "create_time": None, "cpu": None, "mem_mb": None, "clients": None,
        }
        if s["running"]:
            s["https_code"], _ = doctor._get(f"https://localhost:{doctor.HTTPS_PORT}/", timeout=2)
            s["http_code"], _ = doctor._get(f"http://localhost:{doctor.HTTP_PORT}/", timeout=2)
            try:
                import psutil
                p = psutil.Process(pids[0])
                s["create_time"] = p.create_time()
                s["cpu"] = p.cpu_percent(interval=0.2)
                s["mem_mb"] = p.memory_info().rss / 1048576
                clients = 0
                for c in psutil.net_connections(kind="inet"):
                    if c.status == "ESTABLISHED" and c.laddr and c.laddr.port in ports:
                        clients += 1
                s["clients"] = clients
            except Exception:
                pass
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


class Dashboard(QtWidgets.QWidget):
    _log_sig = QtCore.Signal(str)

    NAV = [("grid", "Dashboard"), ("term", "Console"), ("link", "Access"), ("cross", "Doctor")]

    def __init__(self):
        super().__init__()
        self.setObjectName("root")
        self.setWindowTitle("MMG Demo Server — Control Panel")
        self.resize(1200, 720)
        self.setMinimumSize(980, 600)
        self.proc = None
        self._polling = False
        self._url = f"https://localhost:{doctor.HTTPS_PORT}"
        self._tablet_url = self._url
        self._cpu_hist = []

        self._build_ui()
        self._log_sig.connect(self._append_log)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self._toggle_auto(self.auto.isChecked())
        self.refresh()

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
        hdr.addWidget(self.h1); hdr.addStretch(1); hdr.addWidget(self.chip); hdr.addWidget(hbtn)
        right.addLayout(hdr)

        # Pages
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._page_dashboard())
        self.stack.addWidget(self._page_console())
        self.stack.addWidget(self._page_access())
        self.stack.addWidget(self._page_doctor())
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
        h = QtWidgets.QHBoxLayout(card); h.setContentsMargins(16, 14, 16, 14); h.setSpacing(14)
        ic = QtWidgets.QLabel(); ic.setPixmap(_pixmap(icon, "white", 44, bg=color))
        ic.setFixedSize(44, 44)
        tv = QtWidgets.QVBoxLayout(); tv.setSpacing(2)
        val = QtWidgets.QLabel("—"); val.setObjectName("statValue")
        cap = QtWidgets.QLabel(caption); cap.setObjectName("statCaption")
        tv.addWidget(val); tv.addWidget(cap)
        h.addWidget(ic); h.addLayout(tv); h.addStretch(1)
        return card, val

    def _page_dashboard(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(16)

        cards = QtWidgets.QHBoxLayout(); cards.setSpacing(16)
        self.c_status, self.k_status = self._stat_card("status", "STATUS", GREY)
        _, self.k_uptime = self._stat_card_add(cards, "clock", "UPTIME", AZURE)
        _, self.k_clients = self._stat_card_add(cards, "users", "ACTIVE CLIENTS", GREEN)
        _, self.k_cpu = self._stat_card_add(cards, "cpu", "CPU · MEMORY", PURPLE)
        cards.insertWidget(0, self.c_status)
        v.addLayout(cards)

        # Chart + diagnostics side by side
        midrow = QtWidgets.QHBoxLayout(); midrow.setSpacing(16)
        chart_box = QtWidgets.QGroupBox("CPU load — live"); _shadow(chart_box)
        cbl = QtWidgets.QVBoxLayout(chart_box)
        cbl.addWidget(self._build_chart())
        midrow.addWidget(chart_box, 3)

        diag = QtWidgets.QGroupBox("Diagnostics"); _shadow(diag)
        dl = QtWidgets.QVBoxLayout(diag); dl.setSpacing(10)
        self.ports_lbl = self._diag_row(dl, "Ports")
        self.health_lbl = self._diag_row(dl, "Health")
        self.deps_lbl = self._diag_row(dl, "Prereqs")
        dl.addStretch(1)
        midrow.addWidget(diag, 2)
        v.addLayout(midrow, 1)
        return w

    def _stat_card_add(self, row, icon, caption, color):
        card, val = self._stat_card(icon, caption, color)
        row.addWidget(card)
        return card, val

    def _build_chart(self):
        if not HAVE_CHARTS:
            lbl = QtWidgets.QLabel("QtCharts unavailable"); lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.series = None
            return lbl
        self.series = QtCharts.QLineSeries()
        self.area = QtCharts.QAreaSeries(self.series)
        self.area.setPen(QtGui.QPen(QtGui.QColor(AZURE), 2))
        fill = QtGui.QColor(AZURE); fill.setAlpha(50); self.area.setBrush(fill)
        chart = QtCharts.QChart(); chart.addSeries(self.area)
        chart.legend().hide(); chart.setBackgroundVisible(False)
        chart.setMargins(QtCore.QMargins(0, 0, 0, 0))
        self.axX = QtCharts.QValueAxis(); self.axX.setRange(0, 60); self.axX.setLabelsVisible(False)
        self.axX.setGridLineVisible(False)
        self.axY = QtCharts.QValueAxis(); self.axY.setRange(0, 100); self.axY.setLabelFormat("%d%%")
        self.axY.setTickCount(5)
        pen = QtGui.QPen(QtGui.QColor(BORDER)); self.axY.setGridLinePen(pen)
        chart.addAxis(self.axX, QtCore.Qt.AlignBottom); chart.addAxis(self.axY, QtCore.Qt.AlignLeft)
        self.area.attachAxis(self.axX); self.area.attachAxis(self.axY)
        view = QtCharts.QChartView(chart); view.setRenderHint(QtGui.QPainter.Antialiasing)
        view.setMinimumHeight(220); view.setStyleSheet("background:transparent;border:none")
        return view

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

        help_box = QtWidgets.QGroupBox("Protocol cheat-sheet"); _shadow(help_box)
        hl = QtWidgets.QVBoxLayout(help_box)
        tip = QtWidgets.QLabel(
            f"{_dot(GREEN)} <b>https://&lt;host&gt;:{doctor.HTTPS_PORT}</b> — the app<br>"
            f"{_dot(GREEN)} <b>http://&lt;host&gt;:{doctor.HTTP_PORT}</b> — redirects to HTTPS<br>"
            f"{_dot(RED)} http://…:{doctor.HTTPS_PORT} or https://…:{doctor.HTTP_PORT} "
            f"— connection reset")
        tip.setObjectName("url"); tip.setTextFormat(QtCore.Qt.RichText)
        hl.addWidget(tip)
        v.addWidget(help_box); v.addStretch(1)
        return w

    def _page_doctor(self):
        w = QtWidgets.QGroupBox("Diagnostic report"); _shadow(w)
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.doctor_btn = QtWidgets.QPushButton("Run doctor"); self.doctor_btn.setObjectName("primary")
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
        self.start_btn = QtWidgets.QPushButton("Start"); self.start_btn.setObjectName("primary")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.restart_btn = QtWidgets.QPushButton("Restart")
        self.reset_btn = QtWidgets.QPushButton("Reset ports"); self.reset_btn.setObjectName("danger")
        self.start_btn.clicked.connect(self.start_server)
        self.stop_btn.clicked.connect(self.stop_server)
        self.restart_btn.clicked.connect(self.restart_server)
        self.reset_btn.clicked.connect(self.reset_ports)
        for b in (self.start_btn, self.stop_btn, self.restart_btn, self.reset_btn):
            ctl.addWidget(b)
        ctl.addStretch(1)
        self.auto = QtWidgets.QCheckBox("Auto-refresh every"); self.auto.setChecked(True)
        self.auto.toggled.connect(self._toggle_auto)
        self.interval = QtWidgets.QSpinBox()
        self.interval.setRange(1, 60); self.interval.setValue(3); self.interval.setSuffix(" s")
        self.interval.valueChanged.connect(lambda _: self._toggle_auto(self.auto.isChecked()))
        ctl.addWidget(self.auto); ctl.addWidget(self.interval)
        return ctl

    def _diag_row(self, lay, name):
        row = QtWidgets.QHBoxLayout()
        n = QtWidgets.QLabel(name); n.setFixedWidth(66); n.setAlignment(QtCore.Qt.AlignTop)
        n.setStyleSheet(f"color:{GREY};font-weight:700;font-size:12px")
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
        col = GREEN if running else GREY

        self.chip.setText("● RUNNING" if running else "○ STOPPED")
        self.chip.setStyleSheet(
            f"background:{'#e6f7f0' if running else '#eef0f3'};color:{col};"
            f"font-size:12px;font-weight:700;border-radius:13px;padding:5px 14px")

        self.k_status.setText("RUNNING" if running else "STOPPED")
        self.k_status.setStyleSheet(f"font-size:24px;font-weight:800;color:{col}")

        if running and s["create_time"]:
            secs = int(time.time() - s["create_time"])
            self.k_uptime.setText(f"{secs//3600}h {secs%3600//60}m" if secs >= 3600
                                  else f"{secs//60}m {secs%60}s")
        else:
            self.k_uptime.setText("—")

        self.k_clients.setText(str(s["clients"]) if s["clients"] is not None else "—")
        if running and s["cpu"] is not None:
            self.k_cpu.setText(f"{s['cpu']:.0f}% · {s['mem_mb']:.0f}MB")
        else:
            self.k_cpu.setText("—")

        # chart history
        self._push_cpu(s["cpu"] if (running and s["cpu"] is not None) else 0.0)

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

        self.deps_lbl.setText(
            f"{_dot(GREEN if s['dist'] else RED)} dist &nbsp; "
            f"{_dot(GREEN if s['model'] else RED)} model &nbsp; "
            f"{_dot(GREEN if s['bones'] else AMBER)} bones &nbsp; "
            f"{_dot(GREEN if s['cert'] else AMBER)} cert")

        ip = s["ip"]
        self._url = f"https://localhost:{doctor.HTTPS_PORT}"
        self._tablet_url = f"https://{ip}:{doctor.HTTPS_PORT}"
        self.url_lbl.setText(
            f"This device &nbsp;<b>{self._url}</b><br>"
            f"Tablets &nbsp;&nbsp;&nbsp;&nbsp;<b>{self._tablet_url}</b> "
            f"&nbsp;(or http://{ip}:{doctor.HTTP_PORT} — redirects)")

        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _push_cpu(self, cpu):
        self._cpu_hist.append(cpu)
        if len(self._cpu_hist) > 60:
            self._cpu_hist.pop(0)
        if HAVE_CHARTS and self.series is not None:
            self.series.clear()
            for i, v in enumerate(self._cpu_hist):
                self.series.append(i, v)
            self.axX.setRange(0, max(60, len(self._cpu_hist) - 1))

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
            self.reset_ports(silent=True)
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

    def run_doctor(self):
        self.doctor_btn.setEnabled(False)
        self.docout.setPlainText("Running doctor…\n")
        self._dw = DoctorWorker()
        self._dw.done.connect(self._doctor_done)
        self._dw.start()

    def _doctor_done(self, text):
        self.docout.setPlainText(text)
        self.doctor_btn.setEnabled(True)

    # ---- misc ------------------------------------------------------------
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
    app = QtWidgets.QApplication(sys.argv)
    try:
        from qt_material import apply_stylesheet
        apply_stylesheet(app, theme="light_blue.xml")
    except Exception:
        pass
    app.setStyleSheet(app.styleSheet() + QSS)
    win = Dashboard()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
