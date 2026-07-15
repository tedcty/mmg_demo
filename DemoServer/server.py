"""
DemoServer — MMG Outreach Hub

Serves demo.html at / and provides a Python API backend for the SSM Demo.
Supports multiple concurrent tablet sessions — each session gets its own
output files and progress queue so predictions never collide.

Routes:
  GET  /                         → demo.html
  GET  /resources/<path>         → landing-page assets (ABI logo, etc.)
  GET  /posters/<path>           → info/poster PDFs (info-button targets)
  GET  /ssm/                     → SSM Demo (built Vite dist)
  GET  /ssm/<path>               → SSM Demo static assets
  GET  /emg/                     → Spikerbox-EMG browser game (Web Audio)
  GET  /emg/<path>               → EMG game assets (web/ + shared resources/)
  GET  /bones.json               → default mean-model bones.json
  GET  /api/progress?session=X   → SSE stream for session X
  POST /api/predict              → run SSM pipeline for session X
  POST /api/save_report          → save refinement report for session X

Usage:
  conda run -n demo python server.py

First-time build of the SSM frontend (run once from TauriGUI/):
  npm install
  npx vite build --base /ssm/
"""

import os
import re
import sys
import json
import shutil
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
ASSETS_DIR  = os.path.join(BASE_DIR, 'resources')   # landing-page assets (logo, etc.)
POSTERS_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'Documents', 'Posters'))  # info/poster PDFs
SCRIPTS_DIR = os.path.join(SSM_DIR, 'scripts')
RES_DIR     = os.path.join(SSM_DIR, 'Resources')
GUI_DIR     = os.path.join(SSM_DIR, 'TauriGUI')
VITE_DIST   = os.path.join(GUI_DIR, 'dist')
PUBLIC_DIR  = os.path.join(GUI_DIR, 'public')
BONES_JSON  = os.path.join(PUBLIC_DIR, 'bones.json')   # shared mean model
ANTHRO_CSV  = os.path.join(RES_DIR, 'anthro_data.csv')
SSM_MODEL   = os.path.join(RES_DIR, 'SSM_shape_model_103')

# Per-session temp directory (auto-cleaned after SESSION_TTL seconds)
SESSIONS_DIR = os.path.join(BASE_DIR, 'sessions')
SESSION_TTL  = 30 * 60  # 30 minutes
os.makedirs(SESSIONS_DIR, exist_ok=True)

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


@app.route('/posters/<path:path>')
def posters(path):
    """Info/poster PDFs opened from the landing-page info buttons."""
    return send_from_directory(POSTERS_DIR, path)


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
# Startup
# ---------------------------------------------------------------------------

def _startup_init():
    print('=' * 50)
    print('MMG Outreach Demo Server')
    print('=' * 50)

    if not os.path.isdir(VITE_DIST):
        print(f'[WARN] SSM frontend not built. Run from {GUI_DIR}:')
        print('       npm install && npx vite build --base /ssm/')
    else:
        print(f'[OK]   SSM frontend dist found.')

    if not os.path.exists(BONES_JSON):
        print('[INFO] Generating initial bones.json from mean model...')
        try:
            sys.path.insert(0, SCRIPTS_DIR)
            from generate_isb_joints import process_and_export
            process_and_export()
            print('[OK]   Initial bones.json generated.')
        except Exception as e:
            print(f'[WARN] Initial assembly failed: {e}')
    else:
        print('[OK]   bones.json found.')

    # Clean up any leftover session dirs from a previous run
    _cleanup_old_sessions()

    print()
    print('Listening on http://0.0.0.0:8000')
    print('=' * 50)


if __name__ == '__main__':
    _startup_init()
    app.run(host='0.0.0.0', port=8000, threaded=True, debug=False)
