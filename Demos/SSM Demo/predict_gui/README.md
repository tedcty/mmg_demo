# Shoulder Predictor GUI

A consolidated desktop application for predicting shoulder bone geometry from anthropometric measurements and refining joint kinematics.

## Prerequisites

Before setting up, ensure you have the following installed:
- [Node.js](https://nodejs.org/) (v18+)
- [Rust](https://rustup.rs/) (for Tauri backend)
- [Miniconda/Anaconda](https://docs.anaconda.com/free/miniconda/index.html)
- [Conda Environment]: `thmd2` with `vtk`, `pandas`, `sklearn`, `gias3`, and `ptb`.

## Quick Setup

1. Open a terminal in this directory.
2. Run the setup script:
   ```bash
   python setup_project.py
   ```
3. Follow the prompts to install dependencies.

## Running the App

Once setup is complete, launch the application using:
```bash
python run_app.py
```

## Directory Structure

- `TauriGUI/`: Desktop application source (Rust/Vue/Three.js).
- `scripts/`: Python processing engine and kinematic solvers.
- `run_app.py`: Main entry point/launcher.
- `setup_project.py`: Installation helper.
