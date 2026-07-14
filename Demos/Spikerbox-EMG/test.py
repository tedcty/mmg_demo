import queue

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

print(sd.query_devices())

device_id = int(input("Please enter the device id\n"))
fs = 48000
window_seconds = 5  # how many seconds of signal to show on screen

# Thread-safe hand-off: the audio callback runs on its own thread, so we
# can't touch the plot directly from there. Push chunks onto a queue
# instead and let the animation (main thread) drain it.
q = queue.Queue()
buffer = np.zeros(int(fs * window_seconds))


def callback(indata, frames, time, status):
    if status:
        print(status)
    print("Min:", np.min(indata[:, 0]),
          "Max:", np.max(indata[:, 0]))
    q.put(indata[:, 0].copy())


fig, ax = plt.subplots()
time_axis = np.linspace(-window_seconds, 0, len(buffer))
line, = ax.plot(time_axis, buffer)
ax.set_ylim(-1, 1)          # float32 audio is nominally in [-1, 1]
ax.set_xlim(-window_seconds, 0)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Live input signal")


def update_plot(frame):
    global buffer
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        buffer = np.roll(buffer, -shift)
        buffer[-shift:] = data
    line.set_ydata(buffer)
    return (line,)


ani = animation.FuncAnimation(
    fig, update_plot, interval=30, blit=True, cache_frame_data=False
)

stream = sd.InputStream(
    device=device_id,
    samplerate=fs,
    channels=1,
    dtype="float32",
    callback=callback,
)

with stream:
    print("Listening... close the plot window to stop\n")
    plt.show()