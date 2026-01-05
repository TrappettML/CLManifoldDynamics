import jax
import jax.numpy as jnp
from jaxopt import OSQP
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure we use 64-bit precision for stability in geometric projections
jax.config.update("jax_enable_x64", True)

# --- JAX-JIT Compiled Optimization Kernels ---

# FIXED: Added static_argnums for M(2), P(3), N(4) so they can be used for shapes
@jax.jit(static_argnums=(2, 3, 4))
def solve_single_anchor(key, flat_manifolds, M, P, N):
    # 1. Sample direction t and dichotomy y
    key, t_key, y_key = jax.random.split(key, 3)
    
    # CRITICAL FIX: Do NOT normalize t. Keep it standard Normal.
    # Algorithm 2 Step 1: t_k ~ N(0, I_N)
    # N is now static, so this shape creation works
    t = jax.random.normal(t_key, (N,)) 
    
    # P is now static, so this shape creation works
    y = jax.random.choice(y_key, jnp.array([-1.0, 1.0]), (P,))
    
    # 2. Formulate QP
    Q_mat = jnp.eye(N)
    q_vec = -t
    
    # Constraints: y_i * (point_j . x) <= 0
    y_expanded = jnp.repeat(y, M) # (P*M,)
    G_mat = y_expanded[:, None] * flat_manifolds # (P*M, N)
    h_vec = jnp.zeros((P * M,))
    
    # OSQP Solver
    osqp = OSQP(tol=1e-4, maxiter=2000, verbose=False)
    sol = osqp.run(params_obj=(Q_mat, q_vec), params_eq=None, params_ineq=(G_mat, h_vec))
    
    # 3. Recover Anchor Points s_i from Dual Variables
    z = sol.params.dual_ineq
    z = jnp.maximum(z, 0.0)
    
    z_reshaped = z.reshape(P, M)
    z_sums = jnp.sum(z_reshaped, axis=1, keepdims=True)
    
    # Safe division for alpha weights
    safe_sums = z_sums + 1e-10
    alphas = z_reshaped / safe_sums
    
    manifolds = flat_manifolds.reshape(P, M, N)
    
    # s_i is the weighted center of support vectors
    s_i = jnp.einsum('pm,pmn->pn', alphas, manifolds)
    
    # Align sign with dichotomy y
    s_i_aligned = s_i * y[:, None]
    
    return s_i_aligned, t, y


@jax.jit
def compute_metrics_from_anchors(anchors, t_vectors):
    """
    Computes GLUE metrics (Capacity, Radius, Dimension, Alignments) 
    using the Gram Matrix method described in Algorithm 2 and Appendix B.
    
    Args:
        anchors: (n_t, P, N)
        t_vectors: (n_t, N)
    """
    n_t, P, N = anchors.shape
    
    # --- 1. Decomposition into Centers and Axes ---
    
    # Center s^0: Mean over t (Monte Carlo samples)
    s_0 = jnp.mean(anchors, axis=0) # (P, N)
    
    # Axis s^1: Deviation from center
    # s_1 shape: (n_t, P, N)
    s_1 = anchors - s_0[None, :, :]
    
    # --- 2. Gram Matrix Projections per sample k ---
    
    def process_single_sample(s1_k, t_k, anchor_k):
        """
        Process one MC sample k:
        s1_k: (P, N) - Axes for this sample
        t_k: (N,) - Random direction
        anchor_k: (P, N) - Full anchor points (s0 + s1)
        """
        # A. Projection onto Axes Subspace
        # We need || P_axes t ||^2
        # P_axes t is the projection of t onto span(s1_k).
        # Calculated as: v^T (G)^+ v where v = s1_k @ t_k, G = s1_k @ s1_k.T
        
        # Dot products with t: (P,)
        v_axis = s1_k @ t_k 
        # Gram matrix of axes: (P, P)
        G_axis = s1_k @ s1_k.T
        # Pseudo-inverse of Gram matrix (robust inversion)
        G_axis_inv = jnp.linalg.pinv(G_axis, rcond=1e-5)
        
        # Squared norm of projection onto axes
        norm_proj_axis_sq = v_axis.T @ G_axis_inv @ v_axis
        
        # B. Projection onto Total Subspace (Center + Axis)
        v_tot = anchor_k @ t_k
        G_tot = anchor_k @ anchor_k.T
        G_tot_inv = jnp.linalg.pinv(G_tot, rcond=1e-5)
        norm_proj_tot_sq = v_tot.T @ G_tot_inv @ v_tot
        
        return norm_proj_axis_sq, norm_proj_tot_sq

    # Vectorize over n_t samples
    batch_metrics = jax.vmap(process_single_sample)(s_1, t_vectors, anchors)
    norm_proj_axis_sq, norm_proj_tot_sq = batch_metrics
    
    # --- 3. Aggregate Metrics ---
    
    # Manifold Dimension D_M = E[ || proj_axes t ||^2 ]
    D_M = jnp.mean(norm_proj_axis_sq)
    
    # Manifold Radius R_M
    # We will use the robust approximation: R_M = || s_1 ||_F / || s_0 ||_F (averaged)
    # This matches the intuition R ~ noise/signal.
    
    # Let's compute global norms directly from vectors for stability
    norm_s1 = jnp.mean(jnp.sum(s_1**2, axis=-1)) # Mean squared norm of axes
    norm_s0 = jnp.mean(jnp.sum(s_0**2, axis=-1)) # Mean squared norm of centers
    R_M_val = jnp.sqrt(norm_s1 / (norm_s0 + 1e-9))

    # Capacity
    capacity = (1.0 + 1.0 / (R_M_val**2 + 1e-9)) / (D_M + 1e-9)

    # --- 4. Alignment Metrics (GLUE) ---
    
    # Center Alignment (rho_c): correlation between centers
    # 1/(P(P-1)) sum_{i!=j} | <s0_i, s0_j> | / (||s0_i|| ||s0_j||)
    norms_0 = jnp.linalg.norm(s_0, axis=-1, keepdims=True)
    s_0_normed = s_0 / (norms_0 + 1e-9)
    cos_sim_0 = jnp.abs(s_0_normed @ s_0_normed.T)
    # Mask diagonal
    mask = 1.0 - jnp.eye(P)
    rho_c = jnp.sum(cos_sim_0 * mask) / (P * (P - 1))
    
    # Axis Alignment (rho_a)
    # Average correlation between axes of different manifolds
    def compute_axis_corr(s1_k):
        norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
        s1_normed = s1_k / (norms_1 + 1e-9)
        cos_sim_1 = jnp.abs(s1_normed @ s1_normed.T)
        return jnp.sum(cos_sim_1 * mask) / (P * (P - 1))
    
    rho_a = jnp.mean(jax.vmap(compute_axis_corr)(s_1))
    
    # Center-Axis Alignment (psi)
    # Correlation between center of i and axis of j
    def compute_ca_corr(s1_k):
        # s_0_normed: (P, N)
        # s1_k: (P, N)
        norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
        s1_normed = s1_k / (norms_1 + 1e-9)
        # Cross correlation: (P, P) where [i, j] is s0[i] . s1[j]
        cross_sim = jnp.abs(s_0_normed @ s1_normed.T)
        return jnp.sum(cross_sim * mask) / (P * (P - 1))
        
    psi = jnp.mean(jax.vmap(compute_ca_corr)(s_1))
    
    return {
        'Capacity': capacity,
        'Radius': R_M_val,
        'Dimension': D_M,
        'Center_Alignment': rho_c,
        'Axis_Alignment': rho_a,
        'Center_Axis_Alignment': psi
    }

def run_manifold_geometry(representations, labels, n_samples_t=50, seed=42):
    """
    Main entry point for computing manifold metrics.
    
    Args:
        representations: (Total_Samples, Dim) numpy array
        labels: (Total_Samples, ) numpy array
        n_samples_t: Number of Monte Carlo samples for t direction (usually 50-100 is enough)
        seed: Random seed
    """
    # 1. Organize data into manifolds (P, M, N)
    unique_labels = np.unique(labels)
    P = len(unique_labels)
    N = representations.shape[1]
    
    # Determine M (min samples per class to ensure rectangular tensor)
    counts = [np.sum(labels == l) for l in unique_labels]
    M = min(counts)
    
    if M < 2:
        return {k: np.nan for k in ['Capacity', 'Radius', 'Dimension', 
                                   'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']}

    grouped_data = []
    for l in unique_labels:
        # Take first M samples for this class
        idxs = np.where(labels == l)[0][:M]
        grouped_data.append(representations[idxs])
    
    # Stack to (P, M, N)
    manifolds = np.stack(grouped_data)
    
    # JAX arrays
    manifolds_jax = jnp.array(manifolds)
    flat_manifolds = manifolds_jax.reshape(-1, N)
    
    # 2. Batched Optimization (vmap over random keys)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_samples_t)
    
    # We vmap solve_single_anchor over the keys
    # Fix flat_manifolds, M, P, N as non-mapped arguments
    # Since solve_single_anchor has static_argnums=(2,3,4), JAX will look at the 
    # concrete values of M, P, N passed here and compile a specific version for them.
    solve_batch = jax.vmap(solve_single_anchor, in_axes=(0, None, None, None, None))
    
    # Run optimization
    s_aligned, t_vecs, dichotomies = solve_batch(keys, flat_manifolds, M, P, N)
    
    # 3. Compute Metrics
    metrics_jax = compute_metrics_from_anchors(s_aligned, t_vecs)
    
    # Convert to standard python float
    return {k: float(v) for k, v in metrics_jax.items()}

# --- Analysis Pipeline Integration ---

def analyze_manifold_trajectory(config, task_names):
    """
    Loads saved representations and computes manifold metrics over time.
    Plots the results similar to analysis.py.
    """
    print("\n--- Starting Manifold Geometric Analysis (GLUE) ---")
    
    # Metrics to track
    metric_names = ['Capacity', 'Radius', 'Dimension', 
                    'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']
    
    history = {k: [] for k in metric_names}
    task_boundaries = []
    current_epoch = 0
    
    for t_name in task_names:
        rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
        lbl_path = os.path.join(config.reps_dir, f"{t_name}_labels.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(lbl_path):
            print(f"Skipping {t_name} (files not found)")
            continue
            
        reps_data = np.load(rep_path) # (n_steps, n_repeats, n_samples, dim)
        labels = np.load(lbl_path)    # (n_samples, )
        
        if labels.ndim > 1: labels = labels.flatten()
        
        n_steps, n_repeats, n_samples, dim = reps_data.shape
        print(f"Processing {t_name}: {n_steps} steps, {n_repeats} repeats, {dim} dim...")
        
        task_metrics = {k: [] for k in metric_names}
        
        for step in range(n_steps):
            step_res = {k: [] for k in metric_names}
            
            for r in range(n_repeats):
                curr_reps = reps_data[step, r]
                
                # Pre-check for NaNs or Infinite values
                if not np.isfinite(curr_reps).all():
                    for k in metric_names: step_res[k].append(np.nan)
                    continue
                
                try:
                    # Run Analysis with 50 projections
                    res = run_manifold_geometry(curr_reps, labels, n_samples_t=config.n_t)
                    for k in metric_names: step_res[k].append(res[k])
                except Exception as e:
                    print(f"Error in manifold calc at step {step}: {e}")
                    for k in metric_names: step_res[k].append(np.nan)

            for k in metric_names:
                task_metrics[k].append(step_res[k])

        for k in metric_names:
            history[k].extend(task_metrics[k])
            
        current_epoch += n_steps * config.log_frequency
        task_boundaries.append(current_epoch)

    # Convert to arrays
    for k in history:
        if len(history[k]) > 0:
            history[k] = np.array(history[k]) # (Total_Steps, Repeats)

    # --- Plotting ---
    if len(history['Capacity']) == 0:
        print("No manifold metrics computed.")
        return

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    total_steps = len(history[metric_names[0]])
    x_axis = np.arange(1, total_steps + 1) * config.log_frequency
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_metrics))
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = history[metric]
        
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1)
        
        color = colors[i]
        ax.plot(x_axis, mean, label=metric, color=color, linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)
        
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Task Boundaries
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
        if i == 0:
            ax.set_title(f"Manifold Geometric Analysis (GLUE) - {config.dataset_name}")

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, f"manifold_metrics_{config.dataset_name}.png")
    plt.savefig(save_path)
    print(f"Manifold analysis plots saved to {save_path}")