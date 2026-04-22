import sounddevice as sd
import numpy as np

print(sd.query_devices())

device_id = int(input("Please enter the device id\n"))
fs = 48000

def callback(indata, frames, time, status):
    if status:
        print(status)
    indata = indata
    print("Min:", np.min(indata[:, 0]),
          "Max:", np.max(indata[:, 0]))

with sd.InputStream(
    device=device_id,
    samplerate=fs,
    channels=1,
    dtype="float32",
    callback=callback
):
    input("Listening... press Enter to stop\n")