"""
Hierarchical 9-DOF Shoulder Kinematic Constraint Optimizer.

Tree Structure:
  Thorax (Root) -> Clavicle (SC Joint) -> Scapula (AC Joint) -> Humerus (GH Joint)

This implementation follows ISB (Wu et al. 2005) standards for coordinate systems
and utilizes a simultaneous 9-DOF optimization pass to satisfy the Scapulothoracic 
(ST) contact constraint while maintaining anatomical chain connectivity.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


class ShoulderKinematicTree:
    """
    Manages the parent-child hierarchy and recursive transforms for the shoulder.
    
    DOFs (9 total):
    - SC: [Retraction, Elevation, AxialRot] (Y-X-Z)
    - AC: [InternalRot, UpwardRot, Tilt] (Y-X-Z)
    - GH: [Plane, Elev, AxialRot] (Y-X-Y)
    """
    
    def __init__(self, thorax_pts, is_left=False):
        self.is_left = is_left
        self.thorax_center = np.mean(thorax_pts, axis=0)
        self.thorax_pts = thorax_pts
        self.posterior_axis = np.array([0.0, 0.0, 1.0])
        
        # Local Offsets (Fixed relative to bone origins)
        self.sc_pivot_global = None  # Setup during prediction
        self.ac_pivot_local = None   # AC relative to SC local frame
        self.gh_pivot_local = None   # GH relative to AC local frame
        
        # Bone Mesh Prototypes (Zero-rotation pose)
        self.clavicle_mesh = None
        self.scapula_mesh = None
        self.humerus_mesh = None
        
        self.ellipsoid_axes = None
        self.ellipsoid_rot = None
        self._fit_ellipsoid(thorax_pts)

    def _fit_ellipsoid(self, points):
        """Fits a prolate ellipsoid to the filtered thorax vertices."""
        centered = points - self.thorax_center
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        axes_lengths = 2 * np.sqrt(s**2 / max(len(points) - 1, 1))
        self.ellipsoid_rot = vh.T
        axes_sorted = np.sort(axes_lengths)
        a = (axes_sorted[0] + axes_sorted[1]) / 2
        c = axes_sorted[2]
        self.ellipsoid_axes = np.array([a * 0.9, a * 0.9, c * 0.9])

    def get_ellipsoid_distance(self, pts):
        """Normalized squared distance. < 1 = inside."""
        local = (self.ellipsoid_rot.T @ (pts - self.thorax_center).T).T
        return np.sum((local / self.ellipsoid_axes)**2, axis=1)

    def forward_kinematics(self, q):
        """
        Calculates global bone positions based on 9 DOFs.
        q = [scY, scX, scZ, acY, acX, acZ, ghY, ghX, ghY2]
        """
        # 1. SC Joint (Thorax -> Clavicle)
        R_sc = R.from_euler('YXZ', q[0:3], degrees=True).as_matrix()
        clav_transformed = (R_sc @ (self.clavicle_mesh - self.sc_pivot_global).T).T + self.sc_pivot_global
        
        # Update global AC pivot
        ac_pivot_global = (R_sc @ self.ac_pivot_local.T).T + self.sc_pivot_global
        
        # 2. AC Joint (Clavicle -> Scapula)
        R_ac = R_sc @ R.from_euler('YXZ', q[3:6], degrees=True).as_matrix()
        scap_transformed = (R_ac @ (self.scapula_mesh - ac_pivot_global).T).T + ac_pivot_global
        
        # Update global GH pivot
        gh_pivot_global = (R_ac @ self.gh_pivot_local.T).T + ac_pivot_global
        
        # 3. GH Joint (Scapula -> Humerus)
        R_gh = R_ac @ R.from_euler('YXY', q[6:9], degrees=True).as_matrix()
        hum_transformed = (R_gh @ (self.humerus_mesh - gh_pivot_global).T).T + gh_pivot_global
        
        return clav_transformed, scap_transformed, hum_transformed, ac_pivot_global, gh_pivot_global

    def optimize_9dof(self, initial_q, sc_target, ac_target, gh_target):
        """
        Simultaneous optimization of all 9 DOFs.
        Enforces 2mm gap at ST joint and minimizes landmark distance.
        """
        def objective(q):
            clav, scap, hum, ac_p, gh_p = self.forward_kinematics(q)
            
            # 1. Landmark Costs (Centers should stay near predicted points)
            # We allow small drift but penalize > 2mm
            cost_lm = (
                np.linalg.norm(self.sc_pivot_global - sc_target)**2 +
                np.linalg.norm(ac_p - ac_target)**2 +
                np.linalg.norm(gh_p - gh_target)**2
            ) * 5.0
            
            # 2. ST Contact (Target 2mm gap)
            # Use inferior-medial vertices of scapula
            y_thresh = np.percentile(scap[:, 1], 30)
            contact_pts = scap[scap[:, 1] <= y_thresh]
            d = self.get_ellipsoid_distance(contact_pts)
            # Gap cost: penalize if d deviates from 1.0 (surface)
            # 2mm is small compared to rib cage size (~200mm), approx 0.01 in normalized units
            pen_cost = np.sum(np.where(d < 1.0, (1.0 - d)**2, 0.0)) * 500.0 # Strict penetration
            gap_cost = (np.mean(d) - 1.02)**2 * 100.0 # 1.02 is ~2-4mm margin
            
            # 3. Regularization (Stay near initial guess/literature)
            reg_cost = np.sum((q - initial_q)**2) * 0.1
            
            return cost_lm + pen_cost + gap_cost + reg_cost

        bounds = [
            (-30, 30), (0, 30), (-20, 20),   # SC
            (5, 60), (-15, 40), (-25, 25),   # AC
            (-90, 90), (0, 150), (-90, 90)   # GH
        ]
        
        res = minimize(objective, initial_q, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-10})
        return res.x


def solve_hierarchical_shoulder(thorax_pts, bones_mesh_dict, landmarks_dict, is_left=False):
    """
    Entry point for the 9-DOF optimization.
    Handles 'Mirror-to-Right' reflection for the left side.
    """
    if is_left:
        # Mirror all inputs across Sagittal (X=0) plane
        thorax_pts[:, 0] *= -1
        for k in landmarks_dict:
            landmarks_dict[k][0] *= -1
        for k in bones_mesh_dict:
            bones_mesh_dict[k][:, 0] *= -1

    tree = ShoulderKinematicTree(thorax_pts, is_left)
    tree.sc_pivot_global = landmarks_dict['SC']
    tree.clavicle_mesh = bones_mesh_dict['Clavicle']
    tree.scapula_mesh = bones_mesh_dict['Scapula']
    tree.humerus_mesh = bones_mesh_dict['Humerus']
    
    # Pre-calculate local pivots (at zero rotation pose)
    tree.ac_pivot_local = landmarks_dict['AC'] - landmarks_dict['SC']
    tree.gh_pivot_local = landmarks_dict['GH'] - landmarks_dict['AC']
    
    # Seeding: Literature neutral for SC/AC, Neutral for GH
    # Bourne et al 2007: SC(Elev 11, Ret 18), AC(Prot 35, Up 5, Tilt 15)
    initial_q = np.array([
        18.0, 11.0, 0.0,   # SC
        35.0, 5.0, 15.0,   # AC
        0.0, 20.0, 0.0     # GH
    ])
    
    final_q = tree.optimize_9dof(initial_q, landmarks_dict['SC'], 
                                 landmarks_dict['AC'], landmarks_dict['GH'])
    
    clav, scap, hum, ac_p, gh_p = tree.forward_kinematics(final_q)
    
    if is_left:
        # Revert Mirroring
        clav[:, 0] *= -1
        scap[:, 0] *= -1
        hum[:, 0] *= -1
        final_q[0] *= -1 # Sign flip for mirrored protraction etc depends on Euler sequence
        # Note: In a professional model we'd use a more robust reflection mapping,
        # but for visualization, mirroring the geometry is the priority.

    return {
        'Clavicle': clav, 'Scapula': scap, 'Humerus': hum,
        'angles': final_q
    }
