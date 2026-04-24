import numpy as np
import json
import os

def check_reachability():
    json_path = 'E:/Repo/mmg_demo/Demos/SSM Demo/predict_gui/TauriGUI/public/bones.json'
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path) as f:
        data = json.load(f)
    
    right = data['anatomical_landmarks']['right']
    sc = np.array(right['thorax_sc'])
    ac = np.array(right['scapula_ac'])
    aa = np.array(right['scapula_aa'])
    ts = np.array(right['scapula_ts'])
    ai = np.array(right['scapula_ai'])
    
    proj_markers = [m['pos'] for m in data['markers'] if m['label'] == 'R_Proj']
    if not proj_markers:
        print("Error: R_Proj marker not found.")
        return
        
    proj = np.array(proj_markers[0])
    centroid = (aa + ts + ai) / 3.0
    
    L_clav = np.linalg.norm(ac - sc)
    L_scap = np.linalg.norm(centroid - ac)
    D_proj = np.linalg.norm(proj - sc)
    
    min_r = abs(L_clav - L_scap)
    max_r = L_clav + L_scap
    
    print(f"L_clav (SC-AC): {L_clav:.2f}mm")
    print(f"L_scap (AC-Centroid): {L_scap:.2f}mm")
    print(f"D_proj (SC-ProjectedPoint): {D_proj:.2f}mm")
    print(f"Reach Range: [{min_r:.2f}, {max_r:.2f}]mm")
    
    reachable = (min_r <= D_proj <= max_r)
    print(f"Is Reachable via FABRIK: {reachable}")

if __name__ == '__main__':
    check_reachability()
