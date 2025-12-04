import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from typing import List, Dict, Tuple
from ipdb import set_trace

# Suppress CVXOPT verbose output
solvers.options['show_progress'] = False


class GLUEAnalyzer:
    """
    Implements GLUE (Chou et al., 2025a & 2025b) for analysis and visualization.
    """

    @staticmethod
    def solve_qp(X_manifolds: List[np.ndarray], y_dichotomy: np.ndarray, t: np.ndarray):
        """
        Solves the QP to find dual variables (z) for anchor points.
        """
        P = len(X_manifolds)
        dim = X_manifolds[0].shape[1]
        total_points = sum(m.shape[0] for m in X_manifolds)
        
        # Min (1/2)||v - t||^2
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
        """Helper to convert dual vars to anchor vectors."""
        P = len(manifolds)
        anchors = np.zeros((P, dim))
        active_indices_list = []
        
        start_idx = 0
        for mu in range(P):
            m = manifolds[mu]
            m_len = m.shape[0]
            z_slice = z_dual[start_idx : start_idx + m_len]
            start_idx += m_len
            
            sum_z = np.sum(z_slice)
            
            # Identify active points (support vectors)
            # Threshold for numerical stability
            active_idxs = np.where(z_slice > 1e-5)[0]
            active_indices_list.append(active_idxs)

            if sum_z > 1e-9:
                # s^mu = y^mu * (sum z_i x_i) / (sum z_i)
                weighted_sum = z_slice @ m
                anchors[mu] = y_dichotomy[mu] * (weighted_sum / sum_z)
            else:
                anchors[mu] = np.zeros(dim)
                
        return anchors, active_indices_list

    @staticmethod
    def analyze(manifolds: List[np.ndarray], n_samples: int = 100) -> Dict[str, float]:
        """
        Full GLUE analysis returning scalar metrics.
        """
        P = len(manifolds)
        N = manifolds[0].shape[1]
        epsilon = 1e-9
        jitter = 1e-7
        
        history_S = []
        history_t = []
        valid_k = 0
        
        # 1. Sample Anchors
        for k in range(n_samples):
            t = np.random.randn(N)
            # 1-vs-Rest dichotomy sampling
            target_idx = np.random.randint(0, P)
            y = -np.ones(P)
            y[target_idx] = 1.0
            
            z_dual = GLUEAnalyzer.solve_qp(manifolds, y, t)
            if z_dual is None: continue
            
            anchors, _ = GLUEAnalyzer._compute_anchors_from_dual(z_dual, manifolds, y, N)
            history_S.append(anchors)
            history_t.append(t)
            valid_k += 1
            
        if valid_k == 0: return {k: 0.0 for k in ["Capacity", "Radius", "Dimension", "Center Align", "Axis Align", "Center-Axis"]}

        S_stack = np.array(history_S) # (K, P, N)
        t_stack = np.array(history_t) # (K, N)
        
        # 2. Estimate Centers (S0)
        S0 = np.mean(S_stack, axis=0)
        S1_stack = S_stack - S0
        G0 = S0 @ S0.T
        
        # 3. Compute Metrics
        sum_alpha_inv, sum_D, sum_R_num, sum_R_den = 0.0, 0.0, 0.0, 0.0
        sum_rho_a, sum_psi = 0.0, 0.0
        
        for k in range(valid_k):
            Sk = S_stack[k]
            S1k = S1_stack[k]
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
            
            norms_S1 = np.linalg.norm(S1k, axis=1, keepdims=True) + epsilon
            norms_S0 = np.linalg.norm(S0, axis=1, keepdims=True) + epsilon
            
            cos_S1 = (S1k @ S1k.T) / (norms_S1 @ norms_S1.T)
            sum_rho_a += (np.sum(np.abs(cos_S1)) - P)
            
            cos_mix = (S0 @ S1k.T) / (norms_S0 @ norms_S1.T)
            sum_psi += (np.sum(np.abs(cos_mix)) - np.sum(np.abs(np.diag(cos_mix))))

        # Finalize
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
    def get_robust_geometry(H, Y, n_center_samples=30):
        """
        Robustly estimates manifold geometry for visualization.
        
        Returns:
            centers: (P, N) The stable manifold centers (S0).
            sample_anchor: (P, N) A single instance of anchor points (S) for visualization.
            active_points: List[np.ndarray] Support vectors for the sample anchor.
        """
        unique_labels = np.unique(Y)
        if len(unique_labels) < 2: return None, None, None
        
        manifolds = [H[Y.flatten() == lbl] for lbl in unique_labels]
        P = len(manifolds)
        dim = H.shape[1]
        
        # 1. Estimate Centers (S0) by averaging over random t
        # We assume a fixed dichotomy for visualization (0 vs 1)
        y_dichotomy = np.array([-1.0, 1.0]) # Assumes binary for simplicity
        
        accum_anchors = np.zeros((P, dim))
        valid_count = 0
        
        # Monte Carlo estimate of S0
        for _ in range(n_center_samples):
            t_rand = np.random.randn(dim)
            z_dual = GLUEAnalyzer.solve_qp(manifolds, y_dichotomy, t_rand)
            if z_dual is not None:
                anchs, _ = GLUEAnalyzer._compute_anchors_from_dual(z_dual, manifolds, y_dichotomy, dim)
                accum_anchors += anchs
                valid_count += 1
        
        if valid_count == 0: return None, None, None
        centers = accum_anchors / valid_count
        
        # 2. Get one specific sample "Axis" for visualization
        # We pick a random t to show "one instance" of the variation
        t_sample = np.random.randn(dim)
        z_sample = GLUEAnalyzer.solve_qp(manifolds, y_dichotomy, t_sample)
        
        if z_sample is None: return centers, centers, [np.array([]) for _ in range(P)]
        
        sample_anchors, active_indices = GLUEAnalyzer._compute_anchors_from_dual(z_sample, manifolds, y_dichotomy, dim)
        
        active_points = []
        for mu, idxs in enumerate(active_indices):
            if len(idxs) > 0:
                active_points.append(manifolds[mu][idxs])
            else:
                active_points.append(np.empty((0, dim)))
                
        return centers, sample_anchors, active_points