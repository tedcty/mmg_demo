"""
DemoServer — MMG Outreach Hub

Serves demo.html at / and provides a Python API backend for the SSM Demo.
Supports multiple concurrent tablet sessions — each session gets its own
output files and progress queue so predictions never collide.

Routes:
  GET  /                         → demo.html
  GET  /resources/<path>         → landing-page assets (ABI logo, etc.)
  GET  /posters/<path>           → info/poster PDFs (info-button targets)
  GET  /doc-resources/<path>     → shared outreach figures (Documents/resources)
  GET  /trust                    → tablet cert-trust instructions (also on HTTP)
  GET  /rootCA.crt               → the CA to install on a tablet (also on HTTP)
  GET  /ssm/                     → SSM Demo (built Vite dist)
  GET  /ssm/<path>               → SSM Demo static assets
  GET  /emg/                     → Spikerbox-EMG browser game (Web Audio)
  GET  /emg/<path>               → EMG game assets (web/ + shared resources/)
  GET  /segment/                 → Object Segmenter (CT stack)
  GET  /segment/slices/<s>/...   → built CT slices + manifest (build_slices.py)
  GET  /bones.json               → default mean-model bones.json
  GET  /api/progress?session=X   → SSE stream for session X
  POST /api/predict              → run SSM pipeline for session X
  POST /api/save_report          → save refinement report for session X

Usage:
  conda run -n demo python server.py         # HTTPS :8443 (+ http :8000 redirect)
  conda run -n demo python server.py --http  # plain HTTP :8000 (mic only on localhost)
  conda run -n demo python server.py --check # preflight doctor only, then exit

Open http://<host>:8000 (auto-redirects to HTTPS) or https://<host>:8443 directly.

First-time build of the SSM frontend (run once from TauriGUI/):
  npm install
  npx vite build --base /ssm/
"""

import os
import re
import sys
import json
import shutil
import socket
import argparse
import subprocess
import threading
import queue
import time

from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SSM_DIR     = os.path.normpath(os.path.join(BASE_DIR, '..', 'Demos', 'SSM Demo', 'predict_gui'))
EMG_DIR     = os.path.normpath(os.path.join(BASE_DIR, '..', 'Demos', 'Spikerbox-EMG'))
EMG_WEB_DIR = os.path.join(EMG_DIR, 'web')
SEG_DIR     = os.path.normpath(os.path.join(BASE_DIR, '..', 'Demos', 'StrangeObjectSegmenter'))
SEG_WEB_DIR = os.path.join(SEG_DIR, 'web')
# Study data for the segmenter: CT slices built by build_slices.py plus the
# ground-truth bone meshes. Reachable as /segment/slices/… and /segment/mesh/….
SEG_DATA_DIR = os.path.join(SEG_DIR, 'bones')
ASSETS_DIR  = os.path.join(BASE_DIR, 'resources')   # landing-page assets (logo, etc.)
POSTERS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'Documents', 'Posters'))  # info/poster PDFs
DOC_RES_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'Documents', 'resources'))  # shared outreach figures
SCRIPTS_DIR = os.path.join(SSM_DIR, 'scripts')
RES_DIR     = os.path.join(SSM_DIR, 'Resources')
GUI_DIR     = os.path.join(SSM_DIR, 'TauriGUI')
VITE_DIST   = os.path.join(GUI_DIR, 'dist')
PUBLIC_DIR  = os.path.join(GUI_DIR, 'public')
BONES_JSON  = os.path.join(PUBLIC_DIR, 'bones.json')   # shared mean model
ANTHRO_CSV  = os.path.join(RES_DIR, 'anthro_data.csv')
SSM_MODEL   = os.path.join(RES_DIR, 'SSM_shape_model_103')

# TLS cert/key (HTTPS is on by default — see __main__). Drop a trusted pair
# here (e.g. from mkcert) as cert.pem/key.pem; otherwise one is auto-generated.
CERT_DIR  = os.path.join(BASE_DIR, 'certs')
CERT_FILE = os.path.join(CERT_DIR, 'cert.pem')
KEY_FILE  = os.path.join(CERT_DIR, 'key.pem')
# Root CA handed out at /rootCA.crt so tablets can trust the server (copied here
# by setup_https.py). Falls back to the leaf cert if there is no separate CA.
ROOTCA_FILE = os.path.join(CERT_DIR, 'rootCA.pem')
TRUST_HTML  = os.path.join(BASE_DIR, 'trust.html')

# mDNS / Bonjour name — lets tablets reach the server at a fixed
# https://mmg-demo.local:<port> instead of a DHCP IP that can change between
# sessions. Resolved natively by iOS/macOS, Windows 10+ and Linux-with-Avahi;
# the `zeroconf` package bundles its own responder so the host OS needs nothing.
MDNS_HOST = 'mmg-demo'
MDNS_FQDN = f'{MDNS_HOST}.local'


def _rootca_path():
    if os.path.exists(ROOTCA_FILE):
        return ROOTCA_FILE
    if os.path.exists(CERT_FILE):
        return CERT_FILE
    return None

# Per-session temp directory (auto-cleaned after SESSION_TTL seconds)
SESSIONS_DIR = os.path.join(BASE_DIR, 'sessions')
SESSION_TTL  = 30 * 60  # 30 minutes
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Shared EMG-game high-score leaderboard (persisted to a JSON file so it
# survives restarts and is shared by every tablet hitting this server).
EMG_SCORES_FILE = os.path.join(BASE_DIR, 'emg_scores.json')
EMG_TOP_N       = 10

# EMG-game client settings (persisted like the leaderboard).
#   tour_desktop       — the first-run guided tour also shows on desktop browsers
#                        (it always shows once on phones/tablets regardless).
#   tour_reset_on_info — launching the demo via the landing page's info popup
#                        (demo.html) re-triggers the tour even if this browser
#                        already saw it; a direct card click never does.
#   sfx_enabled        — propeller/whoosh/flap/countdown/score sound effects.
#                        Off by default.
#   music_enabled      — background music (title/game/countdown tunes).
#                        Independent of sfx_enabled; on by default.
EMG_CONFIG_FILE = os.path.join(BASE_DIR, 'emg_config.json')
EMG_CONFIG_DEFAULT = {
    'tour_desktop': True, 'tour_reset_on_info': True,
    'sfx_enabled': False, 'music_enabled': True,
}
_emg_config_lock = threading.Lock()
_emg_lock       = threading.Lock()

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
_lock: threading.Lock = threading.Lock()
_sessions: dict[str, queue.Queue] = {}


# Session IDs are used to build filesystem paths, so they must be restricted
# to UUID-safe characters. This blocks directory traversal (e.g. "../../..")
# that would otherwise let a crafted session_id write files outside SESSIONS_DIR.
_SID_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def _valid_sid(sid: str) -> bool:
    return bool(_SID_RE.match(sid or ''))


def _get_queue(sid: str) -> queue.Queue:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = queue.Queue()
        return _sessions[sid]


def _session_dir(sid: str) -> str:
    if not _valid_sid(sid):
        raise ValueError(f'invalid session id: {sid!r}')
    d = os.path.join(SESSIONS_DIR, sid)
    # Defense in depth: ensure the resolved path stays under SESSIONS_DIR.
    root = os.path.realpath(SESSIONS_DIR)
    if os.path.commonpath([os.path.realpath(d), root]) != root:
        raise ValueError(f'session path escapes sessions dir: {sid!r}')
    os.makedirs(d, exist_ok=True)
    return d


def _cleanup_old_sessions() -> None:
    """Remove session dirs (and queues) that haven't been touched in SESSION_TTL seconds."""
    cutoff = time.time() - SESSION_TTL
    for entry in os.scandir(SESSIONS_DIR):
        if entry.is_dir() and entry.stat().st_mtime < cutoff:
            shutil.rmtree(entry.path, ignore_errors=True)
            with _lock:
                _sessions.pop(entry.name, None)


# ---------------------------------------------------------------------------
# Static routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_file(os.path.join(BASE_DIR, 'demo.html'))


@app.route('/resources/<path:path>')
def demo_assets(path):
    """Landing-page static assets (e.g. the ABI logo)."""
    return send_from_directory(ASSETS_DIR, path)


@app.route('/trust')
def trust_page():
    """Instructions for trusting the server's cert on a tablet."""
    if not os.path.exists(TRUST_HTML):
        return 'trust.html missing', 404
    return send_file(TRUST_HTML)


@app.route('/rootCA.crt')
def root_ca():
    """The CA/cert to install on a tablet so HTTPS (and the EMG mic) is trusted."""
    ca = _rootca_path()
    if not ca:
        return 'No certificate available yet — run setup_https.py.', 404
    # This MIME makes iOS/Android offer to install it as a certificate.
    return send_file(ca, mimetype='application/x-x509-ca-cert',
                     as_attachment=True, download_name='rootCA.crt')


@app.route('/posters/<path:path>')
def posters(path):
    """Info/poster PDFs opened from the landing-page info buttons."""
    return send_from_directory(POSTERS_DIR, path)


@app.route('/doc-resources/<path:path>')
def doc_resources(path):
    """Shared outreach figures (hero images, etc.) under Documents/resources."""
    return send_from_directory(DOC_RES_DIR, path)


@app.route('/bones.json')
def serve_bones():
    """Shared mean-model bones — shown before any prediction is run."""
    if not os.path.exists(BONES_JSON):
        return jsonify({'error': 'bones.json not found — run a prediction first'}), 404
    return send_file(BONES_JSON, mimetype='application/json')


@app.route('/ssm/', defaults={'path': 'index.html'})
@app.route('/ssm/<path:path>')
def ssm_app(path):
    if not os.path.isdir(VITE_DIST):
        return (
            '<h2>SSM Demo not built yet.</h2>'
            '<p>Run from <code>TauriGUI/</code>:<br>'
            '<code>npm install &amp;&amp; npx vite build --base /ssm/</code></p>',
            503,
        )
    target = os.path.join(VITE_DIST, path)
    if not os.path.exists(target):
        return send_from_directory(VITE_DIST, 'index.html')
    return send_from_directory(VITE_DIST, path)


@app.route('/emg/', defaults={'path': 'index.html'})
@app.route('/emg/<path:path>')
def emg_app(path):
    """Browser port of the Spikerbox-EMG game (Web Audio + Canvas).

    Serves web/index.html and, for anything under resources/, the shared image
    assets that the desktop app also uses. send_from_directory blocks traversal.
    """
    web_target = os.path.join(EMG_WEB_DIR, path)
    if os.path.exists(web_target):
        return send_from_directory(EMG_WEB_DIR, path)
    # Fall back to the demo folder so `resources/*.png` resolves to the assets
    # shared with main.py (no duplication).
    if os.path.exists(os.path.join(EMG_DIR, path)):
        return send_from_directory(EMG_DIR, path)
    return send_from_directory(EMG_WEB_DIR, 'index.html')


@app.route('/segment/', defaults={'path': 'index.html'})
@app.route('/segment/<path:path>')
def segment_app(path):
    """Object Segmenter — brush-paint segmentation over a CT stack.

    Serves web/, then falls back to bones/ so `slices/<series>/…` resolves to
    the PNGs built by build_slices.py (and `mesh/…` to the ground-truth PLYs).
    If the study hasn't been built the viewer falls back to a synthetic volume,
    so the demo still runs on a checkout with no image data.
    send_from_directory blocks traversal.
    """
    if os.path.exists(os.path.join(SEG_WEB_DIR, path)):
        return send_from_directory(SEG_WEB_DIR, path)
    if os.path.exists(os.path.join(SEG_DATA_DIR, path)):
        return send_from_directory(SEG_DATA_DIR, path)
    return send_from_directory(SEG_WEB_DIR, 'index.html')


# ---------------------------------------------------------------------------
# EMG-game leaderboard  (shared high scores)
# ---------------------------------------------------------------------------

def _load_emg_scores():
    """Read the score list from disk; tolerate a missing/corrupt file."""
    try:
        with open(EMG_SCORES_FILE, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


EMG_DIFFICULTIES = ('easy', 'normal', 'hard')
EMG_KEEP_PER_DIFF = 100   # persisted cap, per difficulty (see _trim_for_storage)


def _group_top(scores, n):
    """Top n scores for each difficulty, highest first, as {diff: [scores]}.
    Scores saved before difficulty was tracked have no `diff` and are left out
    — there's no board for them to belong to."""
    return {
        d: sorted((s for s in scores if s.get('diff') == d),
                  key=lambda s: s.get('score', 0), reverse=True)[:n]
        for d in EMG_DIFFICULTIES
    }


def _trim_for_storage(scores):
    """Cap the persisted file at EMG_KEEP_PER_DIFF scores *per difficulty*
    (instead of one global cap) so a less-played difficulty's scores can't get
    squeezed out of storage by another difficulty's higher ones."""
    kept = []
    for d in EMG_DIFFICULTIES:
        rows = sorted((s for s in scores if s.get('diff') == d),
                       key=lambda s: s.get('score', 0), reverse=True)
        kept.extend(rows[:EMG_KEEP_PER_DIFF])
    return kept


@app.route('/api/emg/scores', methods=['GET'])
def emg_scores_get():
    """Top scores per difficulty, highest first: {easy: [...], normal: [...], hard: [...]}."""
    return jsonify(_group_top(_load_emg_scores(), EMG_TOP_N))


@app.route('/api/emg/scores', methods=['POST'])
def emg_scores_post():
    """Record a score. Returns the updated per-difficulty top lists. Name is
    sanitised to a short label; score is range-checked so a bad client can't
    poison the board. `diff` records which difficulty it was set on — a score
    with no valid difficulty is still saved, but won't appear on any board."""
    data = request.get_json(force=True, silent=True) or {}
    name = re.sub(r'[^A-Za-z0-9 _\-]', '', str(data.get('name', ''))).strip()[:8] or 'Anon'
    try:
        score = int(data.get('score'))
    except (TypeError, ValueError):
        return jsonify({'error': 'invalid score'}), 400
    if not (0 <= score <= 100000):
        return jsonify({'error': 'score out of range'}), 400
    diff = str(data.get('diff', '')).strip().lower()

    with _emg_lock:
        scores = _load_emg_scores()
        entry = {'name': name, 'score': score, 'ts': int(time.time())}
        if diff in EMG_DIFFICULTIES:
            entry['diff'] = diff
        scores.append(entry)
        scores = _trim_for_storage(scores)
        try:
            with open(EMG_SCORES_FILE, 'w', encoding='utf-8') as f:
                json.dump(scores, f)
        except OSError as e:
            return jsonify({'error': f'could not save: {e}'}), 500
        # Flag the row we just created so the client can highlight it without
        # having to guess by name+score (names get sanitised above, and two
        # players can share a name and score). Response-only — not persisted.
        top = _group_top(scores, EMG_TOP_N)
        if diff in EMG_DIFFICULTIES:
            top[diff] = [dict(s, you=True) if s is entry else s for s in top[diff]]
    return jsonify(top)


@app.route('/api/emg/scores', methods=['DELETE'])
def emg_scores_clear():
    """Wipe the shared leaderboard. Returns the (now empty) per-difficulty lists."""
    with _emg_lock:
        try:
            with open(EMG_SCORES_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except OSError as e:
            return jsonify({'error': f'could not clear: {e}'}), 500
    return jsonify(_group_top([], EMG_TOP_N))


def _load_emg_config():
    """Read client settings from disk; tolerate a missing/corrupt file."""
    try:
        with open(EMG_CONFIG_FILE, encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {**EMG_CONFIG_DEFAULT, **data}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return dict(EMG_CONFIG_DEFAULT)


@app.route('/api/emg/config', methods=['GET'])
def emg_config_get():
    """Client settings for the EMG game (e.g. whether the guided tour also
    runs on desktop — the dashboard's Demo Control tab toggles this)."""
    return jsonify(_load_emg_config())


@app.route('/api/emg/config', methods=['PUT'])
def emg_config_put():
    data = request.get_json(force=True, silent=True) or {}
    with _emg_config_lock:
        cfg = _load_emg_config()
        if 'tour_desktop' in data:
            cfg['tour_desktop'] = bool(data['tour_desktop'])
        if 'tour_reset_on_info' in data:
            cfg['tour_reset_on_info'] = bool(data['tour_reset_on_info'])
        if 'sfx_enabled' in data:
            cfg['sfx_enabled'] = bool(data['sfx_enabled'])
        if 'music_enabled' in data:
            cfg['music_enabled'] = bool(data['music_enabled'])
        try:
            with open(EMG_CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
        except OSError as e:
            return jsonify({'error': f'could not save: {e}'}), 500
    return jsonify(cfg)


# ---------------------------------------------------------------------------
# SSE progress stream  (one per session)
# ---------------------------------------------------------------------------

@app.route('/api/progress')
def progress():
    sid = request.args.get('session', '').strip()
    if not _valid_sid(sid):
        return jsonify({'error': 'invalid or missing session query param'}), 400

    q = _get_queue(sid)

    # Drain any stale messages left over from a previous run on this session
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break

    def event_stream():
        while True:
            try:
                msg = q.get(timeout=20)
                if msg is None:
                    break
                yield f'data: {msg}\n\n'
            except queue.Empty:
                yield ': keepalive\n\n'

    return Response(
        event_stream(),
        content_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    sid  = data.get('session_id', '').strip()
    if not _valid_sid(sid):
        return jsonify({'error': 'invalid or missing session_id'}), 400

    _cleanup_old_sessions()

    sess_dir   = _session_dir(sid)
    q          = _get_queue(sid)
    out_ply    = os.path.join(sess_dir, 'predicted_model.ply')
    bones_json = os.path.join(sess_dir, 'bones.json')

    args = {
        'sex':             str(data.get('sex', '0')),
        'age':             str(data.get('age', '0')),
        'height':          str(data.get('height', '0')),
        'weight':          str(data.get('weight', '0')),
        'r_clav_len':      str(data.get('r_clav_len', '0')),
        'r_hum_len':       str(data.get('r_hum_len', '0')),
        'r_hum_epi_width': str(data.get('r_hum_epi_width', '0')),
        'anthro_path':     ANTHRO_CSV,
        'ssm_path':        SSM_MODEL,
        'out_path':        out_ply,
        'export_path':     bones_json,
        'fabrik_step':     int(data.get('fabrik_step', 4)),
    }

    json_args = json.dumps(args)
    script    = os.path.join(SCRIPTS_DIR, 'predict_headless.py')

    q.put('STATUS|Initialising prediction...')

    # Use the current interpreter directly. The server already runs inside the
    # env that has the SSM stack (it imports generate_isb_joints at startup),
    # so sys.executable has every dependency. This avoids depending on `conda`
    # being on PATH — it usually isn't (WinError 2), which 500'd the request.
    proc = subprocess.Popen(
        [sys.executable, script, json_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=SCRIPTS_DIR,
    )

    def _stream_stdout(p):
        for line in p.stdout:
            line = line.strip()
            if line:
                q.put(line)

    t = threading.Thread(target=_stream_stdout, args=(proc,), daemon=True)
    t.start()
    t.join()
    proc.wait()

    if proc.returncode != 0:
        err = proc.stderr.read().strip()
        q.put(f'ERROR|{err}')
        return jsonify({'error': err}), 500

    if not os.path.exists(bones_json):
        return jsonify({'error': 'Pipeline succeeded but bones.json was not written'}), 500

    with open(bones_json) as f:
        bones_data = json.load(f)

    return jsonify(bones_data)


# ---------------------------------------------------------------------------
# Save report endpoint
# ---------------------------------------------------------------------------

@app.route('/api/save_report', methods=['POST'])
def save_report():
    data    = request.get_json(force=True)
    sid     = data.get('session_id', '').strip()
    patient = data.get('patient', {})
    rs      = data.get('right_st', {})
    ls      = data.get('left_st', {})

    if sid and not _valid_sid(sid):
        return jsonify({'error': 'invalid session_id'}), 400

    if sid:
        report_path = os.path.join(_session_dir(sid), 'refinement_report.md')
    else:
        report_path = os.path.join(RES_DIR, 'refinement_report.md')

    report = (
        f"# Shoulder Refinement Report\n\n"
        f"## Patient Information\n"
        f"- **Sex**: {patient.get('sex', '?')}\n"
        f"- **Age**: {patient.get('age', '?')} years\n"
        f"- **Height**: {patient.get('height', '?')} cm\n"
        f"- **Weight**: {patient.get('weight', '?')} kg\n\n"
        f"## Refined Shoulder Kinematics (ISB)\n\n"
        f"### Right Shoulder\n"
        f"- **SC**: Abduction: {float(rs.get('sc_abduction', 0)):.2f}°, "
        f"Elevation: {float(rs.get('sc_elevation', 0)):.2f}°, "
        f"Upward Rot: {float(rs.get('sc_upward', 0)):.2f}°\n"
        f"- **AC**: Internal Rot: {float(rs.get('ac_internal', 0)):.2f}°, "
        f"Upward Rot: {float(rs.get('ac_upward', 0)):.2f}°, "
        f"Posterior Tilt: {float(rs.get('ac_posterior', 0)):.2f}°\n"
        f"- **GH**: Flexion: {float(rs.get('gh_flexion', 0)):.2f}°, "
        f"Abduction: {float(rs.get('gh_abduction', 0)):.2f}°, "
        f"Internal Rot: {float(rs.get('gh_internal', 0)):.2f}°\n\n"
        f"### Left Shoulder\n"
        f"- **SC**: Abduction: {float(ls.get('sc_abduction', 0)):.2f}°, "
        f"Elevation: {float(ls.get('sc_elevation', 0)):.2f}°, "
        f"Upward Rot: {float(ls.get('sc_upward', 0)):.2f}°\n"
        f"- **AC**: Internal Rot: {float(ls.get('ac_internal', 0)):.2f}°, "
        f"Upward Rot: {float(ls.get('ac_upward', 0)):.2f}°, "
        f"Posterior Tilt: {float(ls.get('ac_posterior', 0)):.2f}°\n"
        f"- **GH**: Flexion: {float(ls.get('gh_flexion', 0)):.2f}°, "
        f"Abduction: {float(ls.get('gh_abduction', 0)):.2f}°, "
        f"Internal Rot: {float(ls.get('gh_internal', 0)):.2f}°\n\n"
        f"*Generated by Shoulder Predictor (MMG Outreach Demo)*"
    )

    with open(report_path, 'w') as f:
        f.write(report)

    return f'Report saved to {report_path}'


# ---------------------------------------------------------------------------
# HTTPS  (on by default so tablets can use the microphone in /emg/)
# ---------------------------------------------------------------------------

def _lan_ip():
    """Best-effort primary LAN IP so it can be baked into the cert's SANs."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))   # no packets sent; just picks the route
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


def _start_mdns(ip, port, https=True):
    """Advertise MDNS_FQDN -> ip over mDNS so tablets can use the fixed name.

    Returns the Zeroconf instance (keep a reference — closing it withdraws the
    name) or None if it can't be set up. Never fatal: tablets can still use the
    raw IP if this fails.
    """
    if not ip or ip.startswith('127.'):
        return None
    try:
        from zeroconf import Zeroconf, ServiceInfo
    except Exception:
        print('[WARN] zeroconf not installed — no mmg-demo.local name '
              '(pip install zeroconf). Tablets can still use the IP.')
        return None
    try:
        svc_type = '_https._tcp.local.' if https else '_http._tcp.local.'
        info = ServiceInfo(
            svc_type,
            f'MMG Demo.{svc_type}',
            addresses=[socket.inet_aton(ip)],
            port=port,
            server=f'{MDNS_FQDN}.',          # publishes the A record for the name
            properties={'path': '/'},
        )
        zc = Zeroconf()
        zc.register_service(info)
        print(f'[OK]   Advertising {MDNS_FQDN} -> {ip} over mDNS.')
        return zc
    except Exception as e:
        print(f'[WARN] mDNS advertisement failed ({e}). Tablets can use the IP.')
        return None


def _ensure_cert():
    """Return (cert, key) paths for HTTPS, or None if TLS can't be set up.

    Reuses certs/cert.pem + certs/key.pem if present (drop in a trusted pair
    from mkcert for warning-free iOS/Android). Otherwise auto-generates a
    persistent self-signed cert that includes localhost, 127.0.0.1 and the
    detected LAN IP as SANs, so the same cert works from tablets and can be
    trusted once per device.
    """
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return CERT_FILE, KEY_FILE
    try:
        import ipaddress
        from datetime import datetime, timedelta, timezone
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        ip = _lan_ip()
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        sans = [x509.DNSName('localhost'),
                x509.DNSName(MDNS_FQDN),
                x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'MMG Demo Server')])
        now = datetime.now(timezone.utc)
        cert = (x509.CertificateBuilder()
                .subject_name(name).issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(days=1))
                .not_valid_after(now + timedelta(days=3650))
                .add_extension(x509.SubjectAlternativeName(sans), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .sign(key, hashes.SHA256()))

        os.makedirs(CERT_DIR, exist_ok=True)
        with open(KEY_FILE, 'wb') as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.TraditionalOpenSSL,
                                      serialization.NoEncryption()))
        with open(CERT_FILE, 'wb') as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print(f'[OK]   Generated self-signed cert '
              f'(SANs: localhost, {MDNS_FQDN}, 127.0.0.1, {ip}).')
        return CERT_FILE, KEY_FILE
    except Exception as e:
        print(f'[WARN] Could not set up HTTPS ({e}). Falling back to HTTP.')
        print('       Install `cryptography` (in requirements.txt) for HTTPS.')
        return None


def _start_http_redirect(http_port, https_port):
    """Run a tiny HTTP server that 302-redirects everything to HTTPS.

    Browsers default to http:// when you type `host:port`, which resets against
    the TLS port. This redirector on the plain-HTTP port bounces those requests
    to https://<host>:<https_port> so no one hits ERR_CONNECTION_RESET.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _Redirect(BaseHTTPRequestHandler):
        def _redirect(self):
            host = (self.headers.get('Host', '') or '').split(':')[0] or 'localhost'
            self.send_response(302)
            self.send_header('Location', f'https://{host}:{https_port}{self.path}')
            self.send_header('Content-Length', '0')
            self.end_headers()

        def _send_file(self, path, ctype, download=None):
            try:
                with open(path, 'rb') as f:
                    data = f.read()
            except OSError:
                self.send_response(404)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            if download:
                self.send_header('Content-Disposition', f'attachment; filename="{download}"')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            # The trust page + CA must stay on plain HTTP: before the cert is
            # installed the HTTPS site is untrusted, so redirecting them would
            # be a chicken-and-egg trap. Everything else upgrades to HTTPS.
            path = self.path.split('?', 1)[0]
            if path == '/rootCA.crt':
                ca = _rootca_path()
                if ca:
                    return self._send_file(ca, 'application/x-x509-ca-cert', 'rootCA.crt')
                self.send_response(404)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path in ('/trust', '/trust/'):
                return self._send_file(TRUST_HTML, 'text/html; charset=utf-8')
            self._redirect()

        do_POST = do_PUT = do_DELETE = do_OPTIONS = _redirect

        def do_HEAD(self):
            self._redirect()

        def log_message(self, *args):
            return  # keep the console quiet

    try:
        srv = ThreadingHTTPServer(('0.0.0.0', http_port), _Redirect)
    except OSError as e:
        print(f'[WARN] HTTP->HTTPS redirect not started on :{http_port} ({e}).')
        return
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f'[OK]   HTTP :{http_port} -> HTTPS :{https_port} redirect active.')


def _serve(port, ssl_context):
    """Run the app, keeping Ctrl+C responsive.

    The werkzeug dev server does the TLS handshake on the serving thread, so a
    stuck handshake (seen on locked-down Windows where a security agent probes
    the loopback TLS port) can block Ctrl+C. Run it in a daemon thread and wait
    on the main thread, which stays interruptible; Ctrl+C then always exits.
    """
    import time
    kwargs = dict(host='0.0.0.0', port=port, threaded=True, debug=False)
    if ssl_context is not None:
        kwargs['ssl_context'] = ssl_context
    t = threading.Thread(target=lambda: app.run(**kwargs), daemon=True)
    t.start()
    try:
        while t.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('\nShutting down.')


# ---------------------------------------------------------------------------
# Preflight "doctor"
# ---------------------------------------------------------------------------

def _port_free(port):
    """True if nothing is already listening on the port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('0.0.0.0', port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _doctor(https_port, http_port, want_https):
    """Check the common things that break a run and print a clear report.

    Returns False only on a fatal problem (the port we need is taken); warnings
    don't stop the server.
    """
    print('\n' + '-' * 50)
    print('Preflight check')
    print('-' * 50)
    fatal = False

    def line(label, status, hint=''):
        print(f'  [{status:<4}] {label}' + (f'  - {hint}' if hint else ''))

    # SSM frontend (needed for /ssm/ only)
    if os.path.isdir(VITE_DIST):
        line('SSM frontend built', 'OK')
    else:
        line('SSM frontend built', 'WARN',
             'not built → /ssm/ shows a build message. Run setup_demo_server.py.')

    # SSM prediction inputs
    if os.path.exists(ANTHRO_CSV) and os.path.isdir(SSM_MODEL):
        line('SSM model + data', 'OK')
    else:
        line('SSM model + data', 'WARN', 'Resources/ missing → SSM predictions fail')

    line('bones.json', 'OK' if os.path.exists(BONES_JSON) else 'WARN',
         '' if os.path.exists(BONES_JSON) else 'not generated yet')

    # HTTPS bits
    if want_https:
        if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
            line('TLS cert', 'OK')
        else:
            line('TLS cert', 'INFO', 'will be auto-generated on start')
        if os.path.exists(ROOTCA_FILE):
            line('tablet trust CA', 'OK', 'served at /rootCA.crt (/trust)')
        else:
            line('tablet trust CA', 'INFO',
                 'run setup_https.py for a trusted cert + /trust page')
    else:
        line('mode', 'INFO', 'plain HTTP (--http) — mic only works on localhost')

    # Ports
    main_port = https_port if want_https else http_port
    if _port_free(main_port):
        line(f'port {main_port} free', 'OK')
    else:
        line(f'port {main_port} free', 'FAIL',
             'already in use — stop the other server (close its window) and retry')
        fatal = True
    if want_https and not _port_free(http_port):
        line(f'port {http_port} (redirect)', 'WARN', 'in use → redirect will be skipped')

    print('-' * 50)
    return not fatal


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def _startup_init():
    print('=' * 50)
    print('MMG Outreach Demo Server')
    print('=' * 50)

    # Generate bones.json from the mean model if it isn't there yet (the doctor
    # reports its presence; this is the one step with a side effect).
    if not os.path.exists(BONES_JSON):
        print('[INFO] Generating initial bones.json from mean model...')
        try:
            sys.path.insert(0, SCRIPTS_DIR)
            from generate_isb_joints import process_and_export
            process_and_export()
            print('[OK]   Initial bones.json generated.')
        except Exception as e:
            print(f'[WARN] Initial assembly failed: {e}')

    # Clean up any leftover session dirs from a previous run
    _cleanup_old_sessions()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MMG Outreach Demo Server')
    ap.add_argument('--http', action='store_true',
                    help='serve plain HTTP only, no TLS (mic then works only on '
                         'localhost)')
    ap.add_argument('--https-port', type=int, default=8443,
                    help='HTTPS app port (default 8443)')
    ap.add_argument('--http-port', type=int, default=8000,
                    help='plain-HTTP port: redirects to HTTPS by default, or '
                         'serves the app in --http mode (default 8000)')
    ap.add_argument('--open', action='store_true',
                    help='open the demo in the default browser once serving')
    ap.add_argument('--check', action='store_true',
                    help='run the preflight doctor and exit (no server)')
    args = ap.parse_args()

    _startup_init()

    ssl_context = None if args.http else _ensure_cert()
    ip = _lan_ip()

    ok = _doctor(args.https_port, args.http_port, ssl_context is not None)
    if args.check:
        sys.exit(0 if ok else 1)
    if not ok:
        print('\nPreflight found a fatal problem (see FAIL above). Not starting.')
        sys.exit(1)

    if args.open:
        # Open the DIRECT url (https app port when TLS is on) rather than the
        # http redirect: browsers can auto-upgrade http://localhost to https and
        # then hit the redirect port with TLS → ERR_SSL_PROTOCOL_ERROR. A short
        # delay lets the server bind first so the first load doesn't fail.
        import webbrowser
        if ssl_context:
            url = f'https://localhost:{args.https_port}'
        else:
            url = f'http://localhost:{args.http_port}'
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    print()

    if ssl_context:
        # HTTPS app on 8443, with a plain-HTTP redirector on 8000 so that
        # typing http://host:8000 auto-upgrades instead of resetting.
        _start_http_redirect(args.http_port, args.https_port)
        zc = _start_mdns(ip, args.https_port, https=True)
        print(f'Serving HTTPS on 0.0.0.0:{args.https_port}')
        print(f'  This device : https://localhost:{args.https_port}')
        print(f'  Tablets     : https://{MDNS_FQDN}:{args.https_port}')
        print(f'                https://{ip}:{args.https_port}  (if .local does not resolve)')
        print(f'  (or just open http://<host>:{args.http_port} — it redirects to HTTPS)')
        print(f'  New tablet? open http://{ip}:{args.http_port}/trust to install the cert.')
        print('=' * 50)
        try:
            _serve(args.https_port, ssl_context)
        finally:
            if zc:
                zc.close()
    else:
        zc = _start_mdns(ip, args.http_port, https=False)
        print(f'Serving HTTP on 0.0.0.0:{args.http_port}  (no TLS — mic only on localhost)')
        print(f'  This device : http://localhost:{args.http_port}')
        print(f'  Tablets     : http://{MDNS_FQDN}:{args.http_port}  (mic blocked without HTTPS)')
        print(f'                http://{ip}:{args.http_port}')
        print('=' * 50)
        try:
            _serve(args.http_port, None)
        finally:
            if zc:
                zc.close()
