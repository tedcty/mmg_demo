import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
import time
from pyqtgraph.Qt import QtCore, QtWidgets
from collections import deque
from scipy.signal import butter, sosfilt

# --------------------------------
# Configuration fro data streaming
# --------------------------------
fs = 48000
window_seconds = 4
buffer_size = int(fs * window_seconds)
device_idx = 1
rms_win = 0.02
buffer = deque(maxlen=buffer_size)
cursor_smoothing_seconds = 0.5 # must be larger than RMS window and smaller than window_seconds
cursor_smoothing_samples = int(cursor_smoothing_seconds * fs)

# ----------------------------
# EMG band-pass filter
# ----------------------------
lowcut = 20.0
highcut = 300.0
order = 4

sos = butter(
    order,
    [lowcut, highcut],
    btype='bandpass',
    fs=fs,
    output='sos'
)
# Filter state
zi = np.zeros((sos.shape[0], 2))

# ---------------------------------
# Calibration (resting state & MVC)
# ---------------------------------
calibration_active = False # Boolean about calibration process
calibration_start_time = 0.0
baseline_duration = 2.0  # seconds
mvc_duration = 3.0       # seconds
total_duration = baseline_duration + mvc_duration
baseline_buffer = []
mvc_buffer = []
baseline_value = None
mvc_value = None
calibrate_emg = False # Boolean about calibration applied to EMG data

# ----------------------------
# Audio callback
# ----------------------------
def audio_callback(indata, frames, time, status):
    buffer.extend(indata[:, 0])

stream = sd.InputStream(
    device=device_idx,
    samplerate=fs,
    channels=1,
    dtype='float32',
    callback=audio_callback,
)
stream.start()

# ----------------------------
# Qt Application
# ----------------------------
app = QtWidgets.QApplication(sys.argv)

# Main window widget
main_widget = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(main_widget)

# Button
button = QtWidgets.QPushButton("Calibrate EMG")
layout.addWidget(button)

# Checkbox for calibration (EMG & baseline)
checkbox = QtWidgets.QCheckBox("Use calibration")
layout.addWidget(checkbox)

def toggle_calibration(state):
    global calibrate_emg
    calibrate_emg = state == QtCore.Qt.Checked

    if calibrate_emg:
        # Switch to cursor mode
        curve.setVisible(False)
        cursor.setVisible(True)

        # Normalized EMG → fixed range
        plot.disableAutoRange(axis='y')
        plot.setYRange(0, 1.2)
    else:
        # Switch back to line plot
        cursor.setVisible(False)
        curve.setVisible(True)

        # Absolute EMG → auto range
        plot.enableAutoRange(axis='y')

checkbox.stateChanged.connect(toggle_calibration)

# pyqtgraph widget
win = pg.GraphicsLayoutWidget()
layout.addWidget(win)

# Plot inside pyqtgraph widget
plot = win.addPlot()
plot.getAxis('bottom').setVisible(False)
plot.setLabel('left', 'Amplitude')
plot.enableAutoRange(axis='y')

# EMG line plot
curve = plot.plot(pen='y')
curve.setVisible(True)

# EMG cursor (single moving point)
cursor_x = buffer_size // 2  # fixed x-position (middle of plot)
cursor = pg.ScatterPlotItem(
    x=[cursor_x],
    y=[0],
    size=20,
    brush=pg.mkBrush('y'),
    pen=pg.mkPen(None)
)
cursor.setVisible(False)
plot.addItem(cursor)

main_widget.show()

def start_calibration():
    global calibration_active, calibration_start_time
    global mvc_buffer, baseline_buffer
    baseline_buffer = []
    mvc_buffer = []
    calibration_start_time = time.time()
    calibration_active = True
    print("Calibration started!")

button.clicked.connect(start_calibration)

# ----------------------------
# Update function
# ----------------------------
def update():
    global calibration_active, mvc_value, baseline_value
    global zi

    if buffer:
        data = np.array(buffer)
        # Apply bandpass filter
        filtered, zi = sosfilt(sos, data, zi=zi)
        y = np.zeros(buffer_size)
        y[-len(filtered):] = filtered

        # RMS envelope
        w = max(1, int(rms_win * fs))
        rms = np.sqrt(np.convolve(data ** 2, np.ones(w) / w, mode="valid"))
        y = np.zeros(buffer_size)
        y[-len(rms):] = rms

        # --- Calibration logic ---
        current_emg = y[-1]
        elapsed = time.time() - calibration_start_time
        if calibration_active:
            if elapsed < baseline_duration:
                # Phase 1: baseline (relaxed)
                curve.setPen('b')
                baseline_buffer.append(current_emg)
            elif elapsed < total_duration:
                # Phase 2: MVC (contract)
                curve.setPen('r')
                mvc_buffer.append(current_emg)
            else:
                # Finish calibration
                curve.setPen('y')
                calibration_active = False
                baseline_value = np.mean(baseline_buffer)
                mvc_value = np.mean(mvc_buffer)

                print(f"Baseline RMS: {baseline_value:.4f}")
                print(f"MVC RMS: {mvc_value:.4f}")

        # Normalize EMG data to MVC and correct for baseline
        display_y = y.copy()
        if calibrate_emg and mvc_value is not None and baseline_value is not None:
            emg_corr = display_y - baseline_value
            emg_corr = np.maximum(emg_corr, 0)
            emg_norm = emg_corr / (mvc_value - baseline_value)
            display_y = emg_norm.copy()

            n = min(len(display_y), cursor_smoothing_samples)
            cursor_value = np.mean(display_y[-n:])
            # Update cursor position
            cursor.setData(
                x=[cursor_x],
                y=[cursor_value]
            )
        else:
            # Update line plot
            curve.setData(display_y)


# Timer for real-time updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)  # update every 20 ms

# ----------------------------
# Run
# ----------------------------
sys.exit(app.exec_())