# mmg_demo

Interactive MMG outreach demos — a collection of hands-on musculoskeletal
modelling activities for public engagement, served to tablets from a single
local server (the **DemoServer**, below).

## Demos

### Bone Shape Predictor  (SSM)
[`Demos/SSM Demo`](Demos/SSM%20Demo) · served at `/ssm/`

Enter a few simple measurements (sex, age, height, weight, arm dimensions) and a
**Statistical Shape Model** reconstructs a personalised 3D shoulder. You can then
refine the shoulder joint angles and export a report. Runs in the browser — no
install needed. Also available as a standalone Tauri desktop app.

**Creator:** Ted Yeung, with help from Claude

### Muscles in Control  (EMG)
[`Demos/Spikerbox-EMG`](Demos/Spikerbox-EMG) · served at `/emg/`

Turns live **EMG muscle signals** from a Backyard Brains **SpikerBox** (read over
the device's audio input) into sound and game control. Calibrate by relaxing then
contracting a muscle, then flex to move a bird up and down to catch worms and
dodge cats. Ships as both a desktop app (PyQt5 + pyqtgraph) and a browser version
(Web Audio API) for tablets.

**Creator:** Manuela Zimmer

### Strange Object Segmenter
[`Demos/StrangeObjectSegmenter`](Demos/StrangeObjectSegmenter) · served at `/segment/`

A hands-on **medical-imaging** demo: brush-paint over a foreign object hidden
inside a synthetic image stack, slice by slice, then get a Dice score for how
accurately you traced it — showing how clinicians delineate structures from
imaging data. Fully in-browser and offline (the image stack is generated on the
fly, no assets).

**Creator:** Ted Yeung, with help from Claude

### Surgical Navigation  _(coming soon)_
Interactive demo showing how musculoskeletal modelling helps surgeons navigate
complex anatomy in real time during a procedure.

**Creator:** Originally Ted Yeung, Current version: TBD

### Kinect Motion Capture  _(coming soon)_
Live musculoskeletal kinematics driven by a **Kinect** depth sensor — tracks your
body and animates a musculoskeletal model of your movement in real time.

**Creator:** _TBD_

### Ultrasound Imaging  _(coming soon)_
Shows how **ultrasound** captures muscle and soft tissue in real time, and how
those scans are turned into measurements.

**Creator:** _TBD_

## Demo Server (outreach hub)

The **DemoServer** hosts the outreach landing page and the browser demos (Bone
Shape Predictor at `/ssm/`, Muscles in Control at `/emg/`, Strange Object
Segmenter at `/segment/`) over HTTPS for tablets on a local network.

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
