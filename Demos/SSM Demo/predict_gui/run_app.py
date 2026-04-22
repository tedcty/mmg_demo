import os
import subprocess
import sys

def main():
    # Automatically inject the Rust path into environment variables for the current session
    cargo_path = os.path.join(os.environ.get('USERPROFILE', ''), '.cargo', 'bin')
    os.environ['PATH'] = f"{cargo_path};{os.environ.get('PATH', '')}"
    
    # Locate the Tauri application directory (now a subfolder of this script)
    gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TauriGUI')
    
    print("====================================")
    print("Launching Consolidated Shoulder Predictor...")
    print("  GUI: Tauri / Three.js")
    print("  Logic: scripts/predict_headless.py")
    print("====================================")
    
    # Launch Tauri using npm
    try:
        subprocess.run(["npm", "run", "tauri", "dev"], cwd=gui_dir, shell=True)
    except KeyboardInterrupt:
        print("\nExiting GUI.")
    except Exception as e:
        print(f"\nError launching GUI: {e}")

if __name__ == "__main__":
    main()
