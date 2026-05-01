import numpy as np
import os
import pandas as pd
from gias3.learning import PCA
from sklearn.cross_decomposition import PLSRegression
import vtk
from ptb.util.data import VTKMeshUtl

# plsr

print("--starting PLSR")

np.set_printoptions(threshold=np.inf)

# Inputs
anthro_data = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\PLSR\anthro_data.csv"
case_data = [0, 63, 154.6, 74.7, 170, 296, 55] # Enter patient demographic and anthropometric measurements
ssm_folder_path = r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\SSM\Combined\shape_model"
general_fit_settings = {'ssm_name': 'combinedSSM_mean'}

# Load anthropometric predictors
P = pd.read_csv(anthro_data, header=None)
# predictors_train = P.iloc[:, :15].copy()  # First 15 columns
predictors_train = P.iloc[:, [0, 1, 2, 3, 4, 5, 8, 9]].copy()  # Enter the columns for each anthro data you want to use for prediction
predictors_train.drop([0], axis=0, inplace=True)  # Drop first row (labels)
predictors_train.drop([0], axis=1, inplace=True)  # Drop first column (case ID)

# Load PCA shape model
ssm_pc_file = os.path.join(ssm_folder_path, "combinedSSM.pc.npz")
coupled_pcs = PCA.loadPrincipalComponents(ssm_pc_file)
Y = coupled_pcs.projectedWeights.T

# Run PLS regression
pls2 = PLSRegression(scale=True)
pls2.fit(predictors_train, Y)
pred_weights = pls2.predict([case_data])[0]

# Scale predicted weights
pred_sd = np.zeros((len(pred_weights)))
for j in range(len(pred_weights)):
    pred_sd[j] = pred_weights[j] / np.sqrt(coupled_pcs.weights[j])

print("Predicted SD (first 5):", pred_sd[:5])

# -------------------- Mesh Reconstruction Phase -------------------- #
print("--starting bone prediction")

# Load mean mesh
mean_mesh_file = [f for f in os.listdir(ssm_folder_path) if f.startswith('combinedSSM_mean') and f.endswith('.ply')][0]
mean_mesh_path = os.path.join(ssm_folder_path, mean_mesh_file)

ply_reader = vtk.vtkPLYReader()
ply_reader.SetFileName(mean_mesh_path)
ply_reader.Update()
mesh_data = ply_reader.GetOutput()
mean_mesh_verts = VTKMeshUtl.extract_points(mesh_data)

# Load PC data
pc_file = [f for f in os.listdir(ssm_folder_path) if f.startswith('combinedSSM') and f.endswith('.pc.npz')][0]
pc_file_path = os.path.join(ssm_folder_path, pc_file)
pc = np.load(pc_file_path)

# Calculate scaled weights
pc_modes = pc['modes']
pc_weight = pc['weights']
w_list = np.sqrt(pc_weight)
scaled_weights = (pred_sd * w_list).reshape(1, -1)

# Reconstruct shape
verts = np.dot(scaled_weights, pc_modes.T)
reconstructed_verts = np.reshape(verts, [int(verts.shape[1] / 3), 3]) + mean_mesh_verts
mesh = VTKMeshUtl.update_poly_w_points(reconstructed_verts, mesh_data)

# Save predicted mesh
predicted_path = os.path.join(r"C:\Users\rrag962\Documents\shapemodel\model\Shoulder 3\Segmentations\EOS\2502_predicted.ply")
VTKMeshUtl.write(predicted_path, mesh)

print("--completed prediction")