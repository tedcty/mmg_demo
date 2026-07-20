# mmg_demo

Repository for MMG demos.

## Demo Server (outreach hub)

The **DemoServer** hosts the outreach landing page and the browser demos (SSM
Shoulder Predictor at `/ssm/`, Hear Your Muscles EMG game at `/emg/`) over HTTPS
for tablets on a local network.

```bash
# from the repo root — one-shot setup, then launch
python DemoServer/setup_demo_server.py --run
```

Once running, tablets on the same Wi-Fi open **`https://mmg-demo.local:8443`**
(the server also prints its LAN IP as a fallback). Prefer a window? Run the
control-panel dashboard:

```bash
conda run -n demo python DemoServer/dashboard.py
```

See [`DemoServer/README.md`](DemoServer/README.md) for full setup, HTTPS/cert
trust for tablets, the `mmg-demo.local` mDNS name, and troubleshooting.
