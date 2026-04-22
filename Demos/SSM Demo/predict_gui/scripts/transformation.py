import pandas as pd
from yatpkg.math import transformation
import os
import numpy as np
from scipy.spatial.transform import Rotation

thorax_coords = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\CT_alignment\Thorax\coords3"
scap_coords = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\CT_alignment\scapula\RIGHT\coords5"

thorax_files = [f for f in os.listdir(thorax_coords)]
scap_files = [f for f in os.listdir(scap_coords)]

T = []
for thorax_file in thorax_files:
    case = thorax_file[:7] # change depending on file names
    matching_cases = [scap_file for scap_file in scap_files if scap_file.startswith(case)]
    for matching_case in matching_cases:
        tho = pd.read_csv(os.path.join(thorax_coords, thorax_file))
        org1 = tho['origin'].values
        org_list = org1[0].strip('[]').split()
        org_values = [float(num) for num in org_list]
        org_tho = np.array(org_values)

        tho1 = tho['vectorX'].values
        tho_list = tho1[0].strip('[]').split()
        tho_values = [float(num) for num in tho_list]
        thoX = np.array(tho_values)

        tho2 = tho['vectorY'].values
        tho_list = tho2[0].strip('[]').split()
        tho_values = [float(num) for num in tho_list]
        thoY = np.array(tho_values)

        tho3 = tho['vectorZ'].values
        tho_list = tho3[0].strip('[]').split()
        tho_values = [float(num) for num in tho_list]
        thoZ = np.array(tho_values)

        source = np.array([thoX, thoY, thoZ])

        sca = pd.read_csv(os.path.join(scap_coords, matching_case))
        org1 = sca['origin'].values
        org_list = org1[0].strip('[]').split()
        org_values = [float(num) for num in org_list]
        org_sca = np.array(org_values)

        sca1 = sca['vectorX'].values
        sca_list = sca1[0].strip('[]').split()
        sca_values = [float(num) for num in sca_list]
        scaX = np.array(sca_values)

        sca2 = sca['vectorY'].values
        sca_list = sca2[0].strip('[]').split()
        sca_values = [float(num) for num in sca_list]
        scaY = np.array(sca_values)

        sca3 = sca['vectorZ'].values
        sca_list = sca3[0].strip('[]').split()
        sca_values = [float(num) for num in sca_list]
        scaZ = np.array(sca_values)

        target = np.array([scaX, scaY, scaZ])

        transform_matrix = transformation.Cloud.transform_between_3x3_points_sets(source, target)

        rotate = transform_matrix[:3, :3]
        r = Rotation.from_matrix(rotate)
        r_angles = r.as_euler('xyz', True)
        
        # translation = transform_matrix[:3, 3]

        # distance between thorax and scapula

        translation2 = org_sca - org_tho

        T.append({'case': case, 'sca_Rotate_X': r_angles[0], 'sca_Rotate_Y': r_angles[1],
                       'sca_Rotate_Z': r_angles[2], 'sca_Translate_X': translation2[0],
                       'sca_Translate_Y': translation2[1], 'sca_Translate_Z': translation2[2]})

results = pd.DataFrame(data=T)
out_path = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\CT_alignment\right_transforamtion5.csv"
#save_path = os.path.join(out_path, case + '_Glenoid_inf_angle.csv')
results.to_csv(out_path)