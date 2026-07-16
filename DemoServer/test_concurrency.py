#!/usr/bin/env python3
"""test_concurrency.py — fire N concurrent SSM predictions at the server.

Checks that multiple tablets predicting at once (a) all succeed, (b) get
*isolated* results (no session collisions), and (c) shows the throughput /
latency so you can gauge how many the demo machine handles comfortably.

The other demos (hub, PDFs, EMG game) do no server-side compute — they're just
static files + per-tablet client work — so SSM predict is the only concurrency
bottleneck worth testing.

Run the server in PLAIN-HTTP mode first (this avoids the TLS tooling issues on
locked-down Windows; the :8000 redirector would bounce POSTs otherwise):

    conda run -n demo python server.py --http
    conda run -n demo python test_concurrency.py --n 4

Options:
    --url          server base URL (default http://localhost:8000)
    --n            how many predictions to fire (default 4)
    --concurrency  how many to run at once (default = --n, i.e. all together)
"""

import argparse
import json
import time
import uuid
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

# A plausible case; height/weight are varied per request so each session
# produces a distinct model (lets us verify results are isolated).
BASE_CASE = dict(sex=1, age=30, height=175, weight=75,
                 r_clav_len=150, r_hum_len=330, r_hum_epi_width=60, fabrik_step=4)


def predict(url, i):
    sid = f"loadtest-{uuid.uuid4().hex[:8]}"
    case = dict(BASE_CASE, height=160 + i, weight=65 + i, session_id=sid)
    data = json.dumps(case).encode()
    req = urllib.request.Request(url.rstrip('/') + '/api/predict', data=data,
                                 headers={'Content-Type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.load(resp)
        sig = None
        try:
            sig = tuple(round(x, 2) for x in payload['isb_joints']['right']['gh'])
        except Exception:
            pass
        return dict(i=i, sid=sid, ok=('bones' in payload), dt=time.time() - t0,
                    sig=sig, err=None)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors='replace')[:200]
        hint = ''
        if e.code in (301, 302, 405):
            hint = '  (run the server with --http — this test uses plain HTTP)'
        return dict(i=i, sid=sid, ok=False, dt=time.time() - t0, sig=None,
                    err=f"HTTP {e.code}: {body}{hint}")
    except Exception as e:
        return dict(i=i, sid=sid, ok=False, dt=time.time() - t0, sig=None, err=str(e))


def main():
    ap = argparse.ArgumentParser(description="Concurrent SSM prediction load test.")
    ap.add_argument('--url', default='http://localhost:8000')
    ap.add_argument('--n', type=int, default=4, help='number of predictions')
    ap.add_argument('--concurrency', type=int, default=0,
                    help='parallel workers (default = --n)')
    args = ap.parse_args()
    workers = args.concurrency or args.n

    print(f"Firing {args.n} predictions, {workers} at a time, at {args.url}\n")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(predict, args.url, i) for i in range(args.n)]
        for f in as_completed(futs):
            r = f.result()
            tag = 'OK  ' if r['ok'] else 'FAIL'
            extra = f"err={r['err']}" if r['err'] else f"gh={r['sig']}"
            print(f"  [{tag}] #{r['i']:<2} {r['sid']}  {r['dt']:5.1f}s  {extra}")
            results.append(r)

    total = time.time() - t0
    ok = [r for r in results if r['ok']]
    lat = sorted(r['dt'] for r in results)
    sigs = [r['sig'] for r in ok if r['sig']]
    distinct = len(set(sigs))

    print('-' * 60)
    print(f"  succeeded        : {len(ok)}/{args.n}")
    if lat:
        print(f"  latency          : min {lat[0]:.1f}s  "
              f"median {lat[len(lat)//2]:.1f}s  max {lat[-1]:.1f}s")
    print(f"  wall time        : {total:.1f}s  ({args.n/total:.2f} predictions/s)")
    print(f"  distinct results : {distinct}/{len(sigs)} "
          f"(should equal succeeded -> results are isolated)")
    if len(ok) < args.n:
        print("  RESULT: some predictions FAILED — see err above.")
    elif sigs and distinct < len(sigs):
        print("  RESULT: some results identical — possible session collision.")
    else:
        print("  RESULT: PASS — all succeeded with isolated results.")


if __name__ == '__main__':
    main()
