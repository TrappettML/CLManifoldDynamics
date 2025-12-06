import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from typing import List, Dict, Tuple

# Suppress CVXOPT verbose output
solvers.options['show_progress'] = False

class GLUEAnalyzer:
    """
    Implements GLUE (Chou et al., 2025a & 2025b) for analysis and visualization.
    Ref: Section E.1.4 for sampling and repetition methodology.
    """

    @staticmethod
    def solve_qp(X_manifolds: List[np.ndarray], y_dichotomy: np.ndarray, t: np.ndarray):
        """
        Solves the QP to find dual variables (z) for anchor points.
        Min (1/2)||v - t||^2 s.t. constraints.
        """
        dim = X_manifolds[0].shape[1]
        total_points = sum(m.shape[0] for m in X_manifolds)
        
        P_mat = matrix(np.eye(dim))
        q_mat = matrix(-t)
        
        # Constraints: -y_mu * x_i^T * v <= 0
        G_list = []
        for mu, manifold in enumerate(X_manifolds):
            label = y_dichotomy[mu]
            G_list.append(-label * manifold)
            
        G_mat = matrix(np.vstack(G_list))
        h_mat = matrix(np.zeros(total_points))
        
        try:
            sol = solvers.qp(P_mat, q_mat, G_mat, h_mat)
            if sol['status'] != 'optimal': return None
            return np.array(sol['z']).flatten()
        except:
            return None

    @staticmethod
    def _compute_anchors_from_dual(z_dual, manifolds, y_dichotomy, dim):
        """Helper to convert dual vars to anchor vectors"""
        P = len(manifolds)
        anchors = np.full((P, dim), np.nan) # Initialize with NaN
        
        start_idx = 0
        for mu in range(P):
            m = manifolds[mu]
            m_len = m.shape[0]
            z_slice = z_dual[start_idx : start_idx + m_len]
            start_idx += m_len
            
            sum_z = np.sum(z_slice)
            
            # Identify active points (support vectors) with numerical threshold
            # Use a threshold to avoid numerical instability
            # FIX: increased threshold slightly and checking for valid sum
            if sum_z > 1e-6:
                # Calculate the physical point on the convex hull (weighted sum)
                weighted_sum = z_slice @ m
                anchors[mu] = weighted_sum / sum_z
            # Else: remains NaN
                
        return anchors

    @staticmethod
    def analyze(full_manifolds: List[np.ndarray], 
                n_t_samples: int = 100, 
                n_repeats: int = 10, 
                points_per_sample: int = 50) -> Dict[str, float]:
        """
        Full GLUE analysis with statistical repeats and subsampling.
        Ref: Chou 2025b Section E.1.4.
        """
        metrics_accum = {k: [] for k in ["Capacity", "Radius", "Dimension", "Center Align", "Axis Align", "Center-Axis"]}
        
        for r in range(n_repeats):
            # 1. Subsample Manifolds
            sub_manifolds = []
            for m in full_manifolds:
                n_avail = m.shape[0]
                if n_avail == 0:
                    sub_manifolds.append(m)
                    continue
                idx = np.random.choice(n_avail, min(n_avail, points_per_sample), replace=False)
                sub_manifolds.append(m[idx])
                
            # 2. Run Single Pass Analysis
            res = GLUEAnalyzer._analyze_single_pass(sub_manifolds, n_t_samples)
            if res is not None:
                for k, v in res.items():
                    metrics_accum[k].append(v)
        
        # 3. Average results
        final_metrics = {}
        for k, v_list in metrics_accum.items():
            if len(v_list) > 0:
                final_metrics[k] = np.mean(v_list)
            else:
                final_metrics[k] = 0.0
                
        return final_metrics

    @staticmethod
    def _analyze_single_pass(manifolds: List[np.ndarray], n_samples: int) -> Dict[str, float]:
        """Internal method to compute metrics for one specific subsample  """
        P = len(manifolds)
        N = manifolds[0].shape[1]
        epsilon = 1e-9
        jitter = 1e-7
        
        history_S = []
        history_t = []
        valid_k = 0
        
        for k in range(n_samples):
            t = np.random.randn(N)
            # 1-vs-Rest dichotomy sampling (random target class)
            target_idx = np.random.randint(0, P)
            y = -np.ones(P)
            y[target_idx] = 1.0
            
            z_dual = GLUEAnalyzer.solve_qp(manifolds, y, t)
            if z_dual is None: continue
            
            # Get physical anchors
            anchors = GLUEAnalyzer._compute_anchors_from_dual(z_dual, manifolds, y, N)
            
            # Keep sample if at least one anchor is valid
            if not np.isnan(anchors).all():
                history_S.append(anchors)
                history_t.append(t)
                valid_k += 1
            
        if valid_k == 0: return None

        S_stack = np.array(history_S) # (K, P, N)
        t_stack = np.array(history_t) # (K, N)
        
        # Estimate Centers (S0) ignoring NaNs
        # NanMean ensures we don't drag centers to zero with undefined anchors
        S0 = np.nanmean(S_stack, axis=0) # (P, N)
        
        # If a manifold was NEVER active, S0 will be NaN. Fill with 0 to avoid crash? 
        # Or keep NaN to propagate. For stability, let's fill with 0 but it implies degenerate manifold.
        S0[np.isnan(S0)] = 0.0 
        
        G0 = S0 @ S0.T
        
        sum_alpha_inv, sum_D, sum_R_num, sum_R_den = 0.0, 0.0, 0.0, 0.0
        sum_rho_a, sum_psi = 0.0, 0.0
        
        for k in range(valid_k):
            Sk = S_stack[k].copy()
            # Replace NaN anchors with S0 for metric calculation (zero axis variation)
            # This is a safe fallback for "inactive" manifolds in a sample
            mask_nan = np.isnan(Sk).any(axis=1)
            Sk[mask_nan] = S0[mask_nan]
            
            S1k = Sk - S0
            tk = t_stack[k]
            
            St = Sk @ tk
            G = Sk @ Sk.T
            G_inv = np.linalg.pinv(G + jitter * np.eye(P))
            sum_alpha_inv += St.T @ G_inv @ St
            
            S1t = S1k @ tk
            G1 = S1k @ S1k.T
            G1_inv = np.linalg.pinv(G1 + jitter * np.eye(P))
            term_D = S1t.T @ G1_inv @ S1t
            sum_D += term_D
            
            G_sum = G0 + G1
            G_sum_inv = np.linalg.pinv(G_sum + jitter * np.eye(P))
            term_R_num = S1t.T @ G_sum_inv @ S1t
            sum_R_num += term_R_num
            sum_R_den += np.abs(term_D - term_R_num)
            
            # For alignments, we only sum valid axes
            norms_S1 = np.linalg.norm(S1k, axis=1, keepdims=True) + epsilon
            norms_S0 = np.linalg.norm(S0, axis=1, keepdims=True) + epsilon
            
            cos_S1 = (S1k @ S1k.T) / (norms_S1 @ norms_S1.T)
            sum_rho_a += (np.sum(np.abs(cos_S1)) - P)
            
            cos_mix = (S0 @ S1k.T) / (norms_S0 @ norms_S1.T)
            sum_psi += (np.sum(np.abs(cos_mix)) - np.sum(np.abs(np.diag(cos_mix))))

        capacity = 1.0 / (sum_alpha_inv / (valid_k * P) + epsilon)
        eff_dim = sum_D / (valid_k * P)
        eff_radius = np.sqrt((sum_R_num / valid_k) / (sum_R_den / valid_k + epsilon))
        
        norms_S0 = np.linalg.norm(S0, axis=1, keepdims=True) + epsilon
        cos_S0 = (S0 @ S0.T) / (norms_S0 @ norms_S0.T)
        rho_c = (np.sum(np.abs(cos_S0)) - P) / (P * (P - 1) + epsilon)
        rho_a = sum_rho_a / (valid_k * P * (P - 1) + epsilon)
        psi = sum_psi / (valid_k * P * (P - 1) + epsilon)
        
        return {"Capacity": capacity, "Radius": eff_radius, "Dimension": eff_dim, 
                "Center Align": rho_c, "Axis Align": rho_a, "Center-Axis": psi}

    @staticmethod
    def get_robust_geometry(H, Y, n_center_samples=50, n_visual_anchors=10):
        """
        Robustly estimates manifold geometry for visualization.
        """
        unique_labels = np.unique(Y)
        if len(unique_labels) < 2: return None, None
        
        MAX_PTS = 200
        manifolds = []
        for lbl in unique_labels:
            pts = H[Y.flatten() == lbl]
            if len(pts) > MAX_PTS:
                idx = np.random.choice(len(pts), MAX_PTS, replace=False)
                manifolds.append(pts[idx])
            else:
                manifolds.append(pts)
        
        P = len(manifolds)
        dim = H.shape[1]
        
        y_base = np.array([-1.0, 1.0])
        dichotomies_to_try = [y_base, -y_base]
        
        # Use separate counters to handle per-manifold validity
        accum_anchors = np.zeros((P, dim))
        counts = np.zeros((P, 1))
        
        anchor_history = []
        
        for _ in range(n_center_samples):
            t_rand = np.random.randn(dim)
            
            best_z = None
            best_y = None
            best_total_weight = -1.0
            
            # Find dichotomy that yields the most "active" solution
            for y_try in dichotomies_to_try:
                z_dual = GLUEAnalyzer.solve_qp(manifolds, y_try, t_rand)
                if z_dual is not None:
                    w = np.sum(z_dual)
                    if w > best_total_weight:
                        best_total_weight = w
                        best_z = z_dual
                        best_y = y_try
            
            if best_z is not None and best_total_weight > 1e-6:
                anchs = GLUEAnalyzer._compute_anchors_from_dual(best_z, manifolds, best_y, dim)
                
                # Check which manifolds have valid anchors (not NaN)
                valid_mask = ~np.isnan(anchs).any(axis=1)
                
                if np.any(valid_mask):
                    # Accumulate valid anchors
                    accum_anchors[valid_mask] += anchs[valid_mask]
                    counts[valid_mask] += 1
                    
                    # Store for history (keep NaNs so plotter knows to skip lines)
                    anchor_history.append(anchs)
        
        # Avoid divide by zero
        counts[counts == 0] = 1.0
        centers = accum_anchors / counts
        
        n_avail = len(anchor_history)
        if n_avail == 0: return centers, []
        
        indices = np.random.choice(n_avail, min(n_avail, n_visual_anchors), replace=False)
        anchor_cloud = [anchor_history[i] for i in indices]
                
        return centers, anchor_cloud