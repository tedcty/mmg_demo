# Spikerbox-EMG Demo

A real-time EMG (muscle-signal) demo built with **PyQt5** and **pyqtgraph**.
It reads a Backyard Brains **SpikerBox** over the computer's **audio input**
(via `sounddevice`/PortAudio) and turns your muscle activity into an
interactive game — flex to move the on-screen cursor and collect items.

## Requirements

- **conda** (Miniconda/Anaconda) — the setup script creates an env named `emg`
- **Python 3.12** (installed into the env by the setup script)
- A **SpikerBox** connected to the computer's line-in / microphone input
- **PortAudio** for audio capture — bundled with the `sounddevice` pip wheel;
  on Linux you can also install the system library (e.g. `sudo pacman -S
  portaudio` / `sudo apt install libportaudio2`)

Python deps are pinned in [`requirements.txt`](requirements.txt): numpy, scipy,
sounddevice, pyqtgraph, PyQt5.

## Quick start

```bash
# from this folder (Demos/Spikerbox-EMG)
python setup_emg.py          # create the `emg` conda env + install deps
```

Then connect and power on the SpikerBox and find its audio input index:

```bash
python setup_emg.py --list-devices
```

Note the index of the SpikerBox input. If it is **not `1`**, edit
`device_idx` near the top of [`main.py`](main.py) (~line 27) to match.

Finally, run it:

```bash
python setup_emg.py --run     # launches main.py inside the emg env
```

## Manual steps (equivalent)

```bash
conda create -n emg python=3.12
conda activate emg
pip install -r requirements.txt
python main.py
```

## Configuration

Key settings live near the top of [`main.py`](main.py):

| Setting       | Default | Meaning                                            |
|---------------|---------|----------------------------------------------------|
| `fs`          | 48000   | Audio sample rate (Hz)                             |
| `device_idx`  | 1       | Audio **input** device index (the SpikerBox)      |
| `rms_win`     | 0.02    | RMS smoothing window (s) for the muscle envelope   |

The most common thing to change is `device_idx` — it is hardcoded, so on a new
machine set it to whatever `--list-devices` reports for the SpikerBox.

## Troubleshooting

- **Wrong / silent signal, or an error opening the stream** — `device_idx` is
  pointing at the wrong device. Re-run `--list-devices` and update it.
- **`PortAudioError` / no audio backend** — PortAudio isn't available; install
  the system package (see Requirements).
- **`conda: command not found`** — install Miniconda and reopen your shell.
- **The window doesn't appear (headless/SSH)** — this is a GUI app and needs a
  display; run it on the machine with the monitor, not over a plain SSH session.
