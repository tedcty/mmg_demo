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
from scipy.interpolate import LSQBivariateSpline


def soft_cost(x, delta=10.0):
    """Huber-like cost to prevent gradient explosion."""
    abs_x = np.abs(x)
    return np.where(abs_x < delta, 0.5 * x**2, delta * (abs_x - 0.5 * delta))

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
        
        # Local Offsets (Fixed relative to bone origins)
        self.sc_pivot_global = None  # Setup during prediction
        self.ac_pivot_local = None   # AC relative to SC local frame
        self.gh_pivot_local = None   # GH relative to AC local frame
        
        # Local Scapular Landmarks (Relative to AC pivot)
        self.scap_lms_local = {} 
        
        # Bone Mesh Prototypes (Zero-rotation pose)
        self.clavicle_mesh = None
        self.scapula_mesh = None
        self.humerus_mesh = None
        
        # NURBS-like B-Spline Surface
        self.spline = None
        self._fit_spline(thorax_pts)

    def set_local_landmarks(self, aa, ts, ai, ac_global):
        """Caches landmarks in the scapula local frame (AC = 0,0,0)."""
        self.scap_lms_local = {
            'aa': aa - ac_global,
            'ts': ts - ac_global,
            'ai': ai - ac_global
        }
        print(f"DEBUG: Local landmarks cached. Centroid: {np.mean(list(self.scap_lms_local.values()), axis=0)}")

    def _fit_spline(self, points):
        """Fits a smooth B-Spline surface to the thorax glide area."""
        # Use X, Y as independent variables, Z as the height
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Create knots for the spline
        kx = np.linspace(x.min(), x.max(), 5)
        ky = np.linspace(y.min(), y.max(), 5)
        
        # Fit a least-squares bivariate spline
        self.spline = LSQBivariateSpline(x, y, z, kx[1:-1], ky[1:-1], kx=3, ky=3)
        print(f"DEBUG: NURBS Spline Surface fitted to {len(points)} points.")

    def get_spline_distance(self, pts):
        """Calculates distance from points to the Spline surface."""
        # Target Z = spline(X, Y)
        z_target = self.spline.ev(pts[:, 0], pts[:, 1])
        # Vertical distance is the primary constraint for scapular glide
        return pts[:, 2] - z_target

    def get_spline_normal(self, pts):
        """Calculates the analytical surface normal of the Spline."""
        # Normal is [-df/dx, -df/dy, 1]
        dfdx = self.spline.ev(pts[:, 0], pts[:, 1], dx=1, dy=0)
        dfdy = self.spline.ev(pts[:, 0], pts[:, 1], dx=0, dy=1)
        
        normals = np.zeros((len(pts), 3))
        normals[:, 0] = -dfdx
        normals[:, 1] = -dfdy
        normals[:, 2] = 1.0
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / norms

    def forward_kinematics_6dof(self, q):
        """
        Calculates global bone positions based on 6 DOFs (SC + AC).
        q = [scY, scX, scZ, acY, acX, acZ] (all in degrees)
        """
        # 1. SC Joint (Thorax -> Clavicle)
        # Rotation order YXZ per ISB standards
        R_sc = R.from_euler('YXZ', q[0:3], degrees=True).as_matrix()
        
        # Global AC pivot
        ac_p_global = (R_sc @ self.ac_pivot_local.T).T + self.sc_pivot_global
        
        # 2. AC Joint (Clavicle -> Scapula)
        R_ac_local = R.from_euler('YXZ', q[3:6], degrees=True).as_matrix()
        R_ac_global = R_sc @ R_ac_local
        
        # Transform Scapular Landmarks
        lms_world = {}
        for key, local_vec in self.scap_lms_local.items():
            lms_world[key] = (R_ac_global @ local_vec.T).T + ac_p_global
            
        return lms_world, ac_p_global, R_ac_global

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

    def optimize_6dof(self, initial_q):
        """
        Solves for the 6 DOFs that satisfy the ST contact and normal alignment.
        """
        # Start from Neutral
        initial_q = np.zeros(6) 

        def objective(q):
            lms_world, ac_p, R_ac_global = self.forward_kinematics_6dof(q)
            
            # 1. Scapular Features in World Space
            aa = lms_world['aa']
            ts = lms_world['ts']
            ai = lms_world['ai']
            all_pts = np.array([aa, ts, ai])
            centroid = np.mean(all_pts, axis=0)
            
            # Plane Normal (XS axis per ISB, inverted to point posteriorly)
            v1 = aa - ts
            v2 = ai - ts
            scap_normal = np.cross(v1, v2)
            scap_norm_val = np.linalg.norm(scap_normal)
            if scap_norm_val < 1e-6:
                return 1e12 # Degenerate triangle
            scap_normal /= scap_norm_val
            
            # 2. Thorax Surface Target
            # Distance: z - spline(x,y)
            dists = self.get_spline_distance(all_pts)
            
            # Normalize costs: 
            # Position diff (mm) vs Normal diff (unitless)
            # We want 1mm of error to be roughly equal to 0.01 of normal alignment error
            cost_dist = np.sum(soft_cost(dists, delta=5.0)) * 10.0
            
            # Alignment: (1 - abs(dot)) ranges from 0 to 1
            tho_normal = self.get_spline_normal(centroid.reshape(1,3))[0]
            alignment = np.dot(scap_normal, tho_normal)
            cost_align = (1.0 - abs(alignment))**2 * 500.0
            
            # 3. Regularization (Minimize rotation from neutral)
            cost_reg = np.sum(q**2) * 0.001
            
            return cost_dist + cost_align + cost_reg

        # Initial Debug
        lms_init, _, _ = self.forward_kinematics_6dof(initial_q)
        pts_init = np.array([lms_init['aa'], lms_init['ts'], lms_init['ai']])
        dists_init = self.get_spline_distance(pts_init)
        print(f"DEBUG: Optimization Start. Initial Dists: {dists_init}")
        
        # Joint Limits: Based on anatomical ranges (degrees)
        # SC: Retraction (-20, 40), Elevation (-10, 40), Axial (-30, 30)
        # AC: Internal (0, 60), Upward (-20, 40), Tilt (-30, 30)
        bounds = [
            (-40, 40), (-20, 40), (-30, 30),  # SC
            (-20, 80), (-40, 40), (-40, 40)   # AC
        ]
        
        # Optimization run
        # We'll try L-BFGS-B with tight bounds
        res = minimize(objective, initial_q, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-8})
        
        # If it hits bounds and cost is still high, try a few seeds?
        # But for now, let's see where L-BFGS-B lands.
        
        final_cost = objective(res.x)
        print(f"DEBUG: Optimization End. Final Cost: {final_cost:.2f}")
        return res.x

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
