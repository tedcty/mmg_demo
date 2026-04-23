import os
import subprocess
import sys

def main():
    # Automatically inject the Rust path into environment variables for the current session
    cargo_path = os.path.join(os.environ.get('USERPROFILE', ''), '.cargo', 'bin')
    os.environ['PATH'] = f"{cargo_path};{os.environ.get('PATH', '')}"
    
    # Locate the Tauri application directory (now a subfolder of this script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gui_dir = os.path.join(base_dir, 'TauriGUI')
    
    # Add scripts to path for internal imports
    sys.path.append(os.path.join(base_dir, 'scripts'))
    
    print("====================================")
    print("Launching Consolidated Shoulder Predictor...")
    print("  GUI: Tauri / Three.js")
    print("  Logic: scripts/predict_headless.py")
    print("====================================")
    
    # Run initial assembly to ensure bones.json exists and is correct
    try:
        from scripts.generate_isb_joints import process_and_export
        process_and_export()
    except Exception as e:
        print(f"Warning: Initial assembly failed: {e}")
        print("Continuing to launch GUI...")

    # Launch Tauri using npm
    try:
        subprocess.run(["npm", "run", "tauri", "dev"], cwd=gui_dir, shell=True)
    except KeyboardInterrupt:
        print("\nExiting GUI.")
    except Exception as e:
        print(f"\nError launching GUI: {e}")

if __name__ == "__main__":
    main()
