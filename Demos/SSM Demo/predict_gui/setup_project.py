import os
import subprocess
import sys
import shutil

def run_command(command, cwd=None, shell=None):
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    if shell is None:
        shell = not isinstance(command, list)
    # On Windows, resolve .cmd/.bat shims (npm, yarn, etc.) to a full path
    # since CreateProcess doesn't honor PATHEXT when shell=False.
    if isinstance(command, list) and not shell and os.name == "nt":
        resolved = shutil.which(command[0])
        if resolved:
            command = [resolved, *command[1:]]
    try:
        subprocess.run(command, cwd=cwd, shell=shell, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def setup():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    tauri_dir = os.path.join(root_dir, "TauriGUI")
    
    print("====================================================")
    print("Shoulder Predictor GUI - Setup Utility")
    print("====================================================")

    # 1. Check for NPM and Install Node Modules (Hardened)
    print("\n[1/3] Setting up Node.js dependencies (Secure Mode)...")
    if not shutil.which("npm"):
        print("ERROR: 'npm' not found. Please install Node.js from https://nodejs.org/")
        return

    # Ensure package-lock.json exists for integrity verification
    lock_file = os.path.join(tauri_dir, "package-lock.json")
    if not os.path.exists(lock_file):
        print("WARNING: 'package-lock.json' not found. Hardened 'npm ci' requires a lockfile.")
        print("Falling back to standard 'npm install' - please generate a lockfile for better security.")
        install_cmd = ["npm", "install"]
    else:
        # npm ci is faster, more reliable, and verifies integrity hashes strictly
        print("  Using 'npm ci' for strict dependency integrity verification...")
        install_cmd = ["npm", "ci"]

    if not run_command(install_cmd, cwd=tauri_dir):
        print("ERROR: Failed to install Node.js dependencies.")
        return
    
    print("  Running security audit on dependencies...")
    run_command(["npm", "audit"], cwd=tauri_dir) # Non-blocking audit check
    
    print("SUCCESS: Node.js dependencies installed and verified.")

    # 2. Check for Cargo (Rust)
    print("\n[2/3] Checking for Rust/Cargo...")
    if not shutil.which("cargo"):
        print("WARNING: 'cargo' not found. Rust is required to build the Tauri app.")
        print("Please install Rust from https://rustup.rs/")
    else:
        print("SUCCESS: Rust/Cargo found.")

    # 3. Setup Conda Environment (Optional Step)
    # The SSM stack (numpy, pandas, scipy, scikit-learn, vtk, gias3, ptb_mmg) is
    # shared with the DemoServer, so we reuse the same env name and the single
    # pinned requirements file to keep the two setups from drifting apart.
    env_name = "demo"
    reqs = os.path.normpath(
        os.path.join(root_dir, "..", "..", "..", "DemoServer", "requirements.txt")
    )
    print(f"\n[3/3] Python Environment ('{env_name}')...")

    # Check if conda is available
    if not shutil.which("conda"):
        print(f"WARNING: 'conda' not found. Ensure the '{env_name}' environment exists, then:")
        print(f"  pip install -r {reqs}")
    else:
        print(f"To create or update the environment manually, run:")
        print(f"  conda create -n {env_name} python=3.12")
        print(f"  conda activate {env_name}")
        print(f"  pip install -r {reqs}")

        choice = input(f"\nWould you like to create/update the '{env_name}' environment now? (y/n): ")
        if choice.lower() == 'y':
            # Create the env if it doesn't exist yet (harmless if it already does).
            run_command(["conda", "create", "-y", "-n", env_name, "python=3.12"])
            print(f"Installing pinned dependencies into '{env_name}'...")
            if run_command(["conda", "run", "-n", env_name, "pip", "install", "-r", reqs]):
                print("SUCCESS: Python dependencies installed.")
            else:
                print("ERROR: dependency install failed — see the pip output above.")

    print("\n====================================================")
    print("Setup Complete!")
    print("You can now run the Tauri desktop app using:")
    print(f"  conda activate {env_name}")
    print("  python run_app.py")
    print("\n(For the web Demo Server instead, see DemoServer/README.md)")
    print("====================================================")

if __name__ == "__main__":
    setup()
