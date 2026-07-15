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
# Game definitions
# --------------------------------
game_duration = 30.0 # in seconds
window_seconds = 2 # This defines the x-axis, the smaller the faster the game
coin_chance = 0.7 # Percentage of coins appearing vs. penalties
coin_disappear = 0.6 # Must be >=0.5; Coin or penalty will disappear after % of total time (window_seconds), 0.5 is position of the bird

game_active = False
game_start_time = 0.0

# --------------------------------
# Configuration for data streaming
# --------------------------------
fs = 48000
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
baseline_duration = 3.0  # seconds
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
coin_path = os.path.join(BASE_DIR, "resources", "worm.png")
penalty_path = os.path.join(BASE_DIR, "resources", "cat.png")

bird_y = 0.0
coin_active = penalty_active = False
coin_start_time = penalty_start_time = 0.0
coin_y = penalty_y = 0.0
coin_x = penalty_x = 0.0

X_HIT_THRESHOLD = 50    # horizontal tolerance (to midline)
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

# Start button
button = QtWidgets.QPushButton("Start the game")
top_bar.addWidget(button)

def start_calibration():
    global calibration_active, calibration_start_time, calibrate_emg
    global mvc_buffer, baseline_buffer
    calibrate_emg = False
    toggle_calibration()
    baseline_buffer = []
    mvc_buffer = []
    calibration_start_time = time.time()
    calibration_active = True
    print("Calibration started!")

button.clicked.connect(start_calibration)

def toggle_calibration():
    global calibrate_emg, coin_active, game_active

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
        curve.setVisible(False) # set to True for debugging
        coin_active = False

        # Absolute EMG → auto range
        plot.enableAutoRange(axis='y')

        # Stop the game if it's running
        game_active = False
        game_over_label.hide()

# Play again button
start_button = QtWidgets.QPushButton("Play again")
start_button.setEnabled(False)  # enabled once calibration has completed
top_bar.addWidget(start_button)

def start_game():
    global calibrate_emg, coin_active
    global game_active, game_start_time, score

    calibrate_emg = True
    curve.setVisible(False)
    bird_item.setVisible(True)
    coin_active = True
    plot.disableAutoRange(axis='y')
    plot.setYRange(0, 1.2)

    # Reset score and (re)start the 30s clock
    score = 0
    score_label.setText(f"Score: {score}")
    game_over_label.hide()
    game_start_time = time.time()
    game_active = True

start_button.clicked.connect(start_game)

# Quit button
quit_button = QtWidgets.QPushButton("Quit game")
quit_button.setEnabled(True)
top_bar.addWidget(quit_button)

def quit_game():
    global calibrate_emg, coin_active
    global game_active, game_start_time, score

    calibrate_emg = False
    game_active = False
    curve.setVisible(False)
    game_over_label.hide()
    coin_item.setVisible(False)
    penalty_item.setVisible(False)
    start_button.setEnabled(False)  # enabled once calibration has completed

    # Reset score
    score = 0
    score_label.setText(f"Score: {score}")

quit_button.clicked.connect(quit_game)

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
curve.setVisible(False) # Set to True for debbugging

# EMG cursor (bird image)
bird_img = QtGui.QPixmap(bird_path)
bird_item = QtWidgets.QGraphicsPixmapItem(bird_img)
bird_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
bird_item.setVisible(False)
bird_item.setZValue(100)
bird_item.setOffset(-bird_img.width() / 2, -bird_img.height() / 2) # Remove the /2 for the y coordinate to make the bird stand on the floor

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

# Penalty image
penalty_img = QtGui.QPixmap(penalty_path)
penalty_item = QtWidgets.QGraphicsPixmapItem(penalty_img)
penalty_item.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
penalty_item.setVisible(False)
penalty_item.setZValue(90)
penalty_item.setOffset(-penalty_img.width() / 2, -penalty_img.height() / 2)

win.scene().addItem(penalty_item)

def spawn_penalty():
    global penalty_active, penalty_start_time, penalty_y

    rect = view.sceneBoundingRect()

    penalty_y = rect.top() + np.random.rand() * rect.height()
    penalty_start_time = time.time()
    penalty_active = True

    # Start just outside the right edge
    penalty_item.setPos(rect.right(), penalty_y)
    penalty_item.setVisible(True)

# Game Over overlay
main_rect = main_widget.rect()
game_over_label = QtWidgets.QLabel(main_widget)
game_over_label.setStyleSheet("""
QLabel {
    color: white;
    font-size: 36px;
    font-weight: bold;
    background: rgba(0, 0, 0, 180);
    padding: 20px 40px;
    border-radius: 12px;
}
""")
game_over_label.setAlignment(QtCore.Qt.AlignCenter)
game_over_label.setText(f"Game Over!\nFinal Score: {score}")
game_over_label.adjustSize()
game_over_label.move(
    main_rect.center().x() - game_over_label.width() // 2,
    main_rect.center().y() - game_over_label.height() // 2,
)
game_over_label.hide()

# Relax label (for calibration)
relax_label = QtWidgets.QLabel(main_widget)
relax_label.setStyleSheet("""
QLabel {
    color: white;
    font-size: 36px;
    font-weight: bold;
    background: rgba(0, 0, 0, 180);
    padding: 20px 40px;
    border-radius: 12px;
}
""")
relax_label.setAlignment(QtCore.Qt.AlignCenter)
relax_label.setText("Relax your muscle!")
relax_label.adjustSize()
relax_label.move(
    main_rect.center().x() - relax_label.width() // 2,
    main_rect.center().y() - relax_label.height() // 2,
)
relax_label.hide()

# Contract label (for calibration)
contract_label = QtWidgets.QLabel(main_widget)
contract_label.setStyleSheet("""
QLabel {
    color: white;
    font-size: 36px;
    font-weight: bold;
    background: rgba(0, 0, 0, 180);
    padding: 20px 40px;
    border-radius: 12px;
}
""")
contract_label.setAlignment(QtCore.Qt.AlignCenter)
contract_label.setText("Contract your muscle!")
contract_label.adjustSize()
contract_label.move(
    main_rect.center().x() - contract_label.width() // 2,
    main_rect.center().y() - contract_label.height() // 2,
)
contract_label.hide()

# Rules label
rules_label = QtWidgets.QLabel(main_widget)
rules_label.setStyleSheet("""
QLabel {
    color: white;
    font-size: 18px;
    font-weight: bold;
    background: rgba(0, 0, 0, 180);
    padding: 20px 40px;
    border-radius: 12px;
}
""")
rules_label.setAlignment(QtCore.Qt.AlignCenter)
rules_label.setText("Move the bird up and down\nby contracting and relaxing your muscle!\n\nCatch the worms and avoid the cats!")
rules_label.adjustSize()
rules_label.move(
    main_rect.center().x() - rules_label.width() // 2,
    rules_label.height() // 2
)
rules_label.hide()

def end_game():
    global game_active
    game_active = False

    # Freeze the scene
    bird_item.setVisible(False)
    coin_item.setVisible(False)
    penalty_item.setVisible(False)

    # Show final score, centered over the window
    game_over_label.setText(f"Game Over!\nFinal Score: {score}")
    game_over_label.adjustSize()
    game_over_label.raise_()
    game_over_label.show()

main_widget.show()

# ----------------------------
# Update function
# ----------------------------
def update():
    global calibration_active, mvc_value, baseline_value, calibrate_emg
    global zi
    global bird_y, score, coin_active, coin_x, penalty_active, penalty_x

    if buffer:
        # --- EMG processing ---
        data = np.array(buffer)
        # Apply bandpass filter
        filtered, zi = sosfilt(sos, data, zi=zi)

        # RMS envelope
        w = max(1, int(rms_win * fs))
        rms = np.sqrt(np.convolve(filtered ** 2, np.ones(w) / w, mode="valid"))
        y = np.zeros(buffer_size)
        y[-len(rms):] = rms
        current_emg = y[-1]
        elapsed = time.time() - calibration_start_time

        # --- Calibration logic ---
        if calibration_active:
            rules_label.hide()
            bird_item.setVisible(False)
            coin_item.setVisible(False)
            penalty_item.setVisible(False)

            if elapsed < baseline_duration:
                # Phase 1: baseline (relaxed)
                relax_label.raise_()
                relax_label.show()
                curve.setPen('b')
                baseline_buffer.append(current_emg)
            elif elapsed < total_duration:
                # Phase 2: MVC (contract)
                relax_label.hide()
                contract_label.raise_()
                contract_label.show()
                curve.setPen('r')
                mvc_buffer.append(current_emg)
            else:
                # Finish calibration
                contract_label.hide()
                curve.setPen('y')
                calibration_active = False
                baseline_value = np.mean(baseline_buffer)
                mvc_value = np.mean(mvc_buffer)
                start_button.setEnabled(True)
                calibrate_emg = True
                toggle_calibration() # change display to normalized signal
                start_game() # start the game

                print(f"Baseline RMS: {baseline_value:.4f}")
                print(f"MVC RMS: {mvc_value:.4f}")

        # Grab the data to visualize
        display_y = y.copy()

        # --- End the game ---
        if calibrate_emg and game_active and time.time() - game_start_time >= game_duration:
            end_game()

        # --- Show the rules ---
        if not (game_active or calibration_active or calibrate_emg):
            # Show bird in center
            rect = view.sceneBoundingRect()
            bird_item.setPos(rect.center().x(), rect.center().y())
            coin_item.setPos(rect.center().x() + bird_img.width(), rect.center().y())
            penalty_item.setPos(rect.center().x() - bird_img.width(), rect.center().y())
            bird_item.setVisible(True)
            coin_item.setVisible(True)
            penalty_item.setVisible(True)

            # Show rules
            rules_label.raise_()
            rules_label.show()

        # --- Active game loop ---
        if calibrate_emg and game_active and mvc_value and baseline_value is not None:
            # --- # Normalize EMG data to MVC ---
            emg_corr = display_y - baseline_value
            emg_corr = np.maximum(emg_corr, 0)
            emg_norm = emg_corr / (mvc_value - baseline_value)
            display_y = emg_norm.copy()

            n = min(len(display_y), cursor_smoothing_samples)
            cursor_value = np.mean(display_y[-n:])

            # --- Update bird position ---
            rect = view.sceneBoundingRect()
            bird_x = rect.center().x()
            bird_y = rect.top() + (1 - cursor_value) * rect.height()
            bird_item.setPos(bird_x, bird_y)

            # --- Update flying coin ---
            if coin_active:
                rect = view.sceneBoundingRect()
                elapsed = time.time() - coin_start_time
                t = elapsed / window_seconds

                if t >= 0.6:
                    coin_active = False
                    coin_item.setVisible(False)
                else:
                    x = rect.right() - t * rect.width()
                    coin_x = x
                    coin_item.setPos(coin_x, coin_y)

                    # --- Collision detection ---
                    midline_x = rect.center().x()

                    if abs(coin_x - midline_x) < X_HIT_THRESHOLD and abs(coin_y - bird_y) < Y_HIT_THRESHOLD:
                        # Collision detected
                        coin_active = False
                        coin_item.setVisible(False)
                        score += 1
                        score_label.setText(f"Score: {score}")

            # --- Update flying penalty ---
            if penalty_active:
                rect = view.sceneBoundingRect()
                elapsed = time.time() - penalty_start_time
                t = elapsed / window_seconds

                if t >= 0.6:
                    penalty_active = False
                    penalty_item.setVisible(False)
                else:
                    x = rect.right() - t * rect.width()
                    penalty_x = x
                    penalty_item.setPos(penalty_x, penalty_y)

                    # --- Collision detection ---
                    midline_x = rect.center().x()

                    if abs(penalty_x - midline_x) < X_HIT_THRESHOLD and abs(penalty_y - bird_y) < Y_HIT_THRESHOLD:
                        # Collision detected
                        penalty_active = False
                        penalty_item.setVisible(False)
                        score -= 1
                        score_label.setText(f"Score: {score}")

            if not coin_active and not penalty_active:
                if np.random.rand() < coin_chance:
                    spawn_coin()
                else:
                    spawn_penalty()

        else:
            # Update line plot
            curve.setData(display_y)


# Timer for real-time updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)  # update every 20 ms

# Run application
sys.exit(app.exec_())