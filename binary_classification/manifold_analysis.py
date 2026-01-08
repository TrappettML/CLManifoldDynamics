import jax
import jax.numpy as jnp
from jaxopt import OSQP
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure we use 64-bit precision for stability in geometric projections
jax.config.update("jax_enable_x64", True)

# --- JAX-JIT Compiled Optimization Kernels ---

@jax.jit(static_argnums=(2, 3, 4))
def solve_single_anchor(key, flat_manifolds, M, P, N):
    # 1. Sample direction t and dichotomy y
    key, t_key, y_key = jax.random.split(key, 3)
    
    # t_k ~ N(0, I_N)
    t = jax.random.normal(t_key, (N,)) 
    
    # Random dichotomy y ~ {-1, 1}
    y = jax.random.choice(y_key, jnp.array([-1.0, 1.0]), (P,))
    
    # 2. Formulate QP
    # Obj: min 1/2 ||x - t||^2  => min 1/2 x'Ix - t'x
    Q_mat = jnp.eye(N)
    q_vec = -t
    
    # Constraints: y_i * (point_j . x) >= 0 (Margin condition)
    # Standard QP form in OSQP is: l <= Ax <= u (This wrapper uses Gx <= h)
    # So: -(y_i * point_j) . x <= 0
    y_expanded = jnp.repeat(y, M) # (P*M,)
    
    # FIX 1: Negate G_mat to enforce positive margin (correct classification direction)
    G_mat = -(y_expanded[:, None] * flat_manifolds) # (P*M, N)
    h_vec = jnp.zeros((P * M,))
    
    # OSQP Solver
    osqp = OSQP(tol=1e-5, maxiter=4000, verbose=False)
    sol = osqp.run(params_obj=(Q_mat, q_vec), params_eq=None, params_ineq=(G_mat, h_vec))
    
    # 3. Recover Anchor Points s_i from Dual Variables
    z = sol.params.dual_ineq
    z = jnp.maximum(z, 0.0)
    
    z_reshaped = z.reshape(P, M)
    z_sums = jnp.sum(z_reshaped, axis=1, keepdims=True)
    
    # Normalize duals to get weights (alphas) for each manifold
    safe_sums = z_sums + 1e-10
    alphas = z_reshaped / safe_sums
    
    manifolds = flat_manifolds.reshape(P, M, N)
    
    # s_i_raw: The actual point on the manifold (weighted average of support vectors)
    s_i_raw = jnp.einsum('pm,pmn->pn', alphas, manifolds)
    
    # s_i_aligned: The point flipped to lie in the positive cone (y_i * s_i)
    # Note: We return raw anchors for geometry calculations to avoid zero-mean collapse.
    
    return s_i_raw, t


@jax.jit
def compute_metrics_from_anchors(anchors_raw, t_vectors):
    """
    Computes GLUE metrics.
    
    Args:
        anchors_raw: (n_t, P, N) - The UNALIGNED anchor points on the manifolds.
        t_vectors: (n_t, N)
    """
    n_t, P, N = anchors_raw.shape
    
    # --- 1. Decomposition into Centers and Axes ---
    
    # FIX 2: Use unaligned anchors to compute Centers.
    # If we used aligned anchors, mean would be 0 (due to random y), causing Infinite Radius.
    s_0 = jnp.mean(anchors_raw, axis=0) # (P, N) - Manifold Centroids
    
    # Axis s^1: Deviation from center
    s_1 = anchors_raw - s_0[None, :, :]
    
    # --- 2. Gram Matrix Projections per sample k ---
    
    def process_single_sample(s1_k, t_k):
        """Calculates projection of t onto the axes subspace."""
        v_axis = s1_k @ t_k 
        G_axis = s1_k @ s1_k.T
        
        # Robust pseudo-inverse
        G_axis_inv = jnp.linalg.pinv(G_axis, rcond=1e-5)
        
        norm_proj_axis_sq = v_axis.T @ G_axis_inv @ v_axis
        return norm_proj_axis_sq

    # Vectorize over n_t samples
    norm_proj_axis_sq = jax.vmap(process_single_sample)(s_1, t_vectors)
    
    # --- 3. Aggregate Metrics ---
    
    # Manifold Dimension D_M = E[ || P_axes t ||^2 ]
    D_M = jnp.mean(norm_proj_axis_sq)
    
    # Manifold Radius R_M
    # R_M = Total Variation / Center Norm
    norm_s1 = jnp.mean(jnp.sum(s_1**2, axis=-1)) # Mean squared norm of axes
    norm_s0 = jnp.mean(jnp.sum(s_0**2, axis=-1)) # Mean squared norm of centers
    
    # Avoid div by zero
    R_M_val = jnp.sqrt(norm_s1 / (norm_s0 + 1e-9))

    # Capacity Approximation Formula (Chung et al.)
    capacity = (1.0 + 1.0 / (R_M_val**2 + 1e-9)) / (D_M + 1e-9)

    # --- 4. Alignment Metrics ---
    
    # Center Alignment (rho_c)
    norms_0 = jnp.linalg.norm(s_0, axis=-1, keepdims=True)
    s_0_normed = s_0 / (norms_0 + 1e-9)
    cos_sim_0 = jnp.abs(s_0_normed @ s_0_normed.T)
    mask = 1.0 - jnp.eye(P)
    rho_c = jnp.sum(cos_sim_0 * mask) / (P * (P - 1))
    
    # Axis Alignment (rho_a)
    def compute_axis_corr(s1_k):
        norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
        s1_normed = s1_k / (norms_1 + 1e-9)
        cos_sim_1 = jnp.abs(s1_normed @ s1_normed.T)
        return jnp.sum(cos_sim_1 * mask) / (P * (P - 1))
    
    rho_a = jnp.mean(jax.vmap(compute_axis_corr)(s_1))
    
    # Center-Axis Alignment (psi)
    def compute_ca_corr(s1_k):
        norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
        s1_normed = s1_k / (norms_1 + 1e-9)
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
    """
    # 1. Organize data into manifolds (P, M, N)
    unique_labels = np.unique(labels)
    P = len(unique_labels)
    N = representations.shape[1]
    
    counts = [np.sum(labels == l) for l in unique_labels]
    M = min(counts)
    
    if M < 2:
        return {k: np.nan for k in ['Capacity', 'Radius', 'Dimension', 
                                   'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']}

    grouped_data = []
    for l in unique_labels:
        idxs = np.where(labels == l)[0][:M]
        grouped_data.append(representations[idxs])
    
    manifolds = np.stack(grouped_data)
    manifolds_jax = jnp.array(manifolds)
    flat_manifolds = manifolds_jax.reshape(-1, N)
    
    # 2. Batched Optimization
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_samples_t)
    
    solve_batch = jax.vmap(solve_single_anchor, in_axes=(0, None, None, None, None))
    
    # Returns unaligned anchors
    s_raw, t_vecs = solve_batch(keys, flat_manifolds, M, P, N)
    
    # 3. Compute Metrics
    metrics_jax = compute_metrics_from_anchors(s_raw, t_vecs)
    
    return {k: float(v) for k, v in metrics_jax.items()}

# --- Analysis Pipeline Integration ---

def analyze_manifold_trajectory(config, task_names):
    """
    Loads saved representations and computes manifold metrics over time.
    Returns:
        full_results (dict): A dictionary keyed by task_name.
                             Each value is a dict of metrics: {metric: np.array(steps, repeats)}
    """
    print("\n--- Starting Manifold Geometric Analysis (GLUE) ---")
    
    metric_names = ['Capacity', 'Radius', 'Dimension', 
                    'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']
    
    # Dictionary to store full un-averaged results per task
    full_results = {}
    
    # For plotting aggregate history
    plot_history = {k: [] for k in metric_names}
    task_boundaries = []
    current_epoch = 0
    
    for t_name in task_names:
        rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
        lbl_path = os.path.join(config.reps_dir, f"{t_name}_labels.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(lbl_path):
            print(f"Skipping {t_name} (files not found)")
            continue
            
        reps_data = np.load(rep_path) 
        labels = np.load(lbl_path)    
        
        if labels.ndim > 1: labels = labels.flatten()
        
        n_steps, n_repeats, n_samples, dim = reps_data.shape
        n_steps, n_repeats, n_samples, dim = reps_data.shape
        
        # The labels were flattened in single_run.py: (Samples * Repeats,)
        # We must reshape them to (Samples, Repeats) to extract the correct column per repeat.
        if labels.size == n_samples * n_repeats:
            labels_reshaped = labels.reshape(n_samples, n_repeats)
        else:
            # Fallback for edge cases (e.g., if labels weren't flattened or dim=1)
            print(f"Warning: Label shape mismatch (Got {labels.shape}, expected ({n_samples}, {n_repeats})). utilizing raw labels.")
            labels_reshaped = labels.reshape(n_samples, -1) 
        
        print(f"Processing {t_name}: {n_steps} steps, {n_repeats} repeats, {dim} dim...")
        
        # Storage for current task: metric -> list of steps (where each step is list of repeats)
        task_metrics_lists = {k: [] for k in metric_names}
        
        for step in range(n_steps):
            step_res = {k: [] for k in metric_names}
            
            for r in range(n_repeats):
                curr_reps = reps_data[step, r]
                curr_labels = labels_reshaped[:, r]
                
                if not np.isfinite(curr_reps).all():
                    for k in metric_names: step_res[k].append(np.nan)
                    continue
                
                try:
                    res = run_manifold_geometry(curr_reps, curr_labels, n_samples_t=config.n_t)
                    for k in metric_names: step_res[k].append(res[k])
                except Exception as e:
                    print(f"Error in manifold calc at step {step}: {e}")
                    for k in metric_names: step_res[k].append(np.nan)

            for k in metric_names:
                task_metrics_lists[k].append(step_res[k])

        # Convert lists to arrays (Steps, Repeats) and store in full_results
        full_results[t_name] = {}
        for k in metric_names:
            arr_data = np.array(task_metrics_lists[k]) # Shape: (Steps, Repeats)
            full_results[t_name][k] = arr_data
            plot_history[k].extend(task_metrics_lists[k]) # Extend for plotting

        current_epoch += n_steps * config.log_frequency
        task_boundaries.append(current_epoch)

    # Convert plot history to numpy for plotting
    for k in plot_history:
        if len(plot_history[k]) > 0:
            plot_history[k] = np.array(plot_history[k]) 

    # --- Plotting ---
    if len(plot_history['Capacity']) == 0:
        print("No manifold metrics computed.")
        return full_results

    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    total_steps = len(plot_history[metric_names[0]])
    x_axis = np.arange(1, total_steps + 1) * config.log_frequency
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_metrics))
    
    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = plot_history[metric] # (Total_Steps, Repeats)
        
        if data.size == 0: continue
            
        mean = np.nanmean(data, axis=1)
        std = np.nanstd(data, axis=1)
        
        color = colors[i]
        ax.plot(x_axis, mean, label=metric, color=color, linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)
        
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
        if i == 0:
            ax.set_title(f"Manifold Geometric Analysis (GLUE) - {config.dataset_name}")

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, f"manifold_metrics_{config.dataset_name}.png")
    plt.savefig(save_path)
    print(f"Manifold analysis plots saved to {save_path}")
    
    return full_results