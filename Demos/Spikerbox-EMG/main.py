import os
import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
import time
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from collections import deque
from scipy.signal import butter, sosfilt

# --------------------------------
# Configuration for data streaming
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

# ---------------------------------
# Visuals
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
bg_path = os.path.join(BASE_DIR, "resources", "background.png")
bird_path = os.path.join(BASE_DIR, "resources", "bird.png")
coin_path = os.path.join(BASE_DIR, "resources", "coin.png")

coin_active = False
coin_start_time = 0.0
coin_duration = window_seconds  # seconds
coin_y = 0.0
coin_x = 0.0
bird_y = 0.0

X_HIT_THRESHOLD = 50    # how close to midline
Y_HIT_THRESHOLD = 30    # vertical tolerance

score = 0

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
main_widget.resize(1000, 700)

# Background Image
bg_img = QtGui.QPixmap(bg_path)
bg_label = QtWidgets.QLabel(main_widget)
bg_label.setPixmap(bg_img)
bg_label.setScaledContents(True)
bg_label.setGeometry(main_widget.rect())
bg_label.lower()  # send to back

# Foreground widgets
foreground = QtWidgets.QWidget(main_widget)
foreground.setGeometry(main_widget.rect())
foreground.raise_()  # bring to front
foreground.setStyleSheet("background: transparent;")

# Layout
layout = QtWidgets.QVBoxLayout(foreground)
layout.setContentsMargins(10, 10, 10, 10)
layout.setSpacing(8)
top_bar = QtWidgets.QHBoxLayout()
top_bar.setSpacing(12)

# Button
button = QtWidgets.QPushButton("Calibrate EMG")
top_bar.addWidget(button)

def start_calibration():
    global calibration_active, calibration_start_time
    global mvc_buffer, baseline_buffer
    baseline_buffer = []
    mvc_buffer = []
    calibration_start_time = time.time()
    calibration_active = True
    print("Calibration started!")

button.clicked.connect(start_calibration)

# Checkbox for calibration (EMG & baseline)
checkbox = QtWidgets.QCheckBox("Use calibration")
top_bar.addWidget(checkbox)

def toggle_calibration(state):
    global calibrate_emg, coin_active
    calibrate_emg = state == QtCore.Qt.Checked

    if calibrate_emg:
        # Switch to cursor mode
        curve.setVisible(False)
        bird_item.setVisible(True)
        coin_active = True

        # Normalized EMG → fixed range
        plot.disableAutoRange(axis='y')
        plot.setYRange(0, 1.2)
    else:
        # Switch back to line plot
        bird_item.setVisible(False)
        curve.setVisible(True)
        coin_active = False

        # Absolute EMG → auto range
        plot.enableAutoRange(axis='y')

checkbox.stateChanged.connect(toggle_calibration)

# Score display
score_label = QtWidgets.QLabel("Score: 0")
score_label.setStyleSheet("""
QLabel {
    color: white;
    font-size: 20px;
    font-weight: bold;
    background: rgba(0, 0, 0, 120);
    padding: 6px 10px;
    border-radius: 6px;
}
""")
top_bar.addStretch()
top_bar.addWidget(score_label)

layout.addLayout(top_bar)

# EMG Plot
win = pg.GraphicsLayoutWidget()
layout.addWidget(win)

plot = win.addPlot()
plot.getAxis('bottom').setVisible(False)
plot.getAxis('left').setVisible(False)
plot.enableAutoRange(axis='y')

# Make pyqtgraph fully transparent
win.setBackground(None)
plot.getViewBox().setBackgroundColor(None)

# EMG line plot
curve = plot.plot(pen='y')
curve.setVisible(True)

# EMG cursor (bird image)
bird_img = QtGui.QPixmap(bird_path)
bird_item = QtWidgets.QGraphicsPixmapItem(bird_img)
bird_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
bird_item.setVisible(False)
bird_item.setZValue(100)
bird_item.setOffset(-bird_img.width() / 2, -bird_img.height()) # This offsets the height of the bird such that bird is standing on the floor at 0

view = plot.getViewBox()
win.scene().addItem(bird_item)

# Coin image
coin_img = QtGui.QPixmap(coin_path)
coin_item = QtWidgets.QGraphicsPixmapItem(coin_img)
coin_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
coin_item.setVisible(False)
coin_item.setZValue(90)
coin_item.setOffset(-coin_img.width() / 2, -coin_img.height() / 2)

win.scene().addItem(coin_item)

def spawn_coin():
    global coin_active, coin_start_time, coin_y

    rect = view.sceneBoundingRect()

    coin_y = rect.top() + np.random.rand() * rect.height()
    coin_start_time = time.time()
    coin_active = True

    # Start just outside the right edge
    coin_item.setPos(rect.right(), coin_y)
    coin_item.setVisible(True)

main_widget.show()

# ----------------------------
# Update function
# ----------------------------
def update():
    global calibration_active, mvc_value, baseline_value
    global zi
    global coin_active, coin_x, bird_y, score

    if buffer:
        data = np.array(buffer)
        # Apply bandpass filter
        filtered, zi = sosfilt(sos, data, zi=zi)

        # RMS envelope
        w = max(1, int(rms_win * fs))
        rms = np.sqrt(np.convolve(filtered ** 2, np.ones(w) / w, mode="valid"))
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

            # Update bird position
            rect = view.sceneBoundingRect()
            bird_x = rect.center().x()
            bird_y = rect.top() + (1 - cursor_value) * rect.height()

            bird_item.setPos(bird_x, bird_y)

            # --- Update flying coin ---
            if coin_active:
                rect = view.sceneBoundingRect()
                elapsed = time.time() - coin_start_time
                t = elapsed / coin_duration

                if t >= 1.0:
                    coin_active = False
                    coin_item.setVisible(False)
                else:
                    x = rect.right() - t * rect.width()
                    coin_x = x
                    coin_item.setPos(coin_x, coin_y)

                    # --- Collision detection ---
                    rect = view.sceneBoundingRect()
                    midline_x = rect.center().x()

                    if abs(coin_x - midline_x) < X_HIT_THRESHOLD and abs(coin_y - bird_y) < Y_HIT_THRESHOLD:
                        # Collision detected
                        coin_active = False
                        coin_item.setVisible(False)
                        score += 1
                        score_label.setText(f"Score: {score}")

            if not coin_active:
                spawn_coin()

        else:
            # Update line plot
            curve.setData(display_y)



# Timer for real-time updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)  # update every 20 ms

# Run application
sys.exit(app.exec_())