import os
import subprocess
import sys
import shutil

def run_command(command, cwd=None, shell=True):
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
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
    print("\n[3/3] Python Environment (thmd2)...")
    env_name = "thmd2"
    
    # Check if conda is available
    if not shutil.which("conda"):
        print("WARNING: 'conda' not found. You will need to manually ensure the 'thmd2' environment exists.")
    else:
        print(f"To create or update the environment, run:")
        print(f"  conda create -n {env_name} python=3.10")
        print(f"  conda activate {env_name}")
        print(f"  pip install numpy pandas scikit-learn vtk gias3 ptb")
        
        choice = input(f"\nWould you like to try creating/updating the '{env_name}' environment now? (y/n): ")
        if choice.lower() == 'y':
            print(f"Updating {env_name}...")
            # This assumes the user has access to the required packages
            run_command(f"conda install -n {env_name} -y numpy pandas scikit-learn vtk")
            print("\nNote: 'gias3' and 'ptb' often require manual installation or specific wheels.")

    print("\n====================================================")
    print("Setup Complete!")
    print("You can now run the application using:")
    print("  python run_app.py")
    print("====================================================")

if __name__ == "__main__":
    setup()
