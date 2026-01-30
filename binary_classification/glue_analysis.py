import jax
import jax.numpy as jnp
from jaxopt import OSQP
import matplotlib.pyplot as plt
import os



class glue_solver():
    def __init__(self, glue_key, point_clouds: int, m_points: int, n_dim_space: int, n_t: int, make_plots: bool):
        self.P = point_clouds
        self.M = m_points
        self.N = n_dim_space
        self.n_t = n_t
        self.A = jnp.eye(self.N)
        self.H = jnp.zeros((self.P*self.M,1))
        # set up probes and dichotomies
        y_dichotomies = None
        t_mu = jnp.zeros(self.N)
        t_sigma = jnp.eye(self.N)

        # use glue key to generate t and y keys
        t_key, y_key = jax.random.split(glue_key)
        self.all_t_ks = jax.random.multivariate_normal(t_key, t_mu, t_sigma, shape=(self.n_t,))
        self.all_y_ks = None


    def sample_single_anchor_point(self):
        pass

# --- JAX-JIT Compiled Optimization Kernels ---

@jax.jit(static_argnums=(2, 3, 4))
def solve_single_anchor(key, flat_manifolds, M, P, N):
    """
    Solves for a single anchor point (s_i) and direction (t).
    Pure JAX implementation of the dual SVM problem.
    """
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
    
    # Negate G_mat to enforce positive margin
    G_mat = -(y_expanded[:, None] * flat_manifolds) # (P*M, N)
    h_vec = jnp.zeros((P * M,))
    
    # OSQP Solver
    solver = OSQP(tol=1e-5, maxiter=4000, verbose=False)
    sol = solver.run(params_obj=(Q_mat, q_vec), params_eq=None, params_ineq=(G_mat, h_vec))
    
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
    
    return s_i_raw, t


@jax.jit
def compute_metrics_from_anchors(anchors_raw, t_vectors):
    """
    Computes GLUE metrics from the solved anchor points.
    """
    n_t, P, N = anchors_raw.shape
    
    # --- 1. Decomposition into Centers and Axes ---
    s_0 = jnp.mean(anchors_raw, axis=0) # (P, N) - Manifold Centroids
    
    # Axis s^1: Deviation from center
    s_1 = anchors_raw - s_0[None, :, :]
    
    # --- 2. Gram Matrix Projections per sample k ---
    def process_single_sample(s1_k, t_k):
        v_axis = s1_k @ t_k 
        G_axis = s1_k @ s1_k.T
        G_axis_inv = jnp.linalg.pinv(G_axis, rcond=1e-5)
        norm_proj_axis_sq = v_axis.T @ G_axis_inv @ v_axis
        return norm_proj_axis_sq

    norm_proj_axis_sq = jax.vmap(process_single_sample)(s_1, t_vectors)
    
    # --- 3. Aggregate Metrics ---
    D_M = jnp.mean(norm_proj_axis_sq)
    
    norm_s1 = jnp.mean(jnp.sum(s_1**2, axis=-1)) # Mean squared norm of axes
    norm_s0 = jnp.mean(jnp.sum(s_0**2, axis=-1)) # Mean squared norm of centers
    
    R_M_val = jnp.sqrt(norm_s1 / (norm_s0 + 1e-9))

    # Capacity Approximation Formula (Chung et al.)
    def compute_capacity(r, d):
        return (1.0 + 1.0 / (r**2)) / d
    
    def return_nan(r, d):
        return jnp.nan
    
    # Check if metrics are valid (not too small, not NaN)
    is_valid = (R_M_val > 1e-6) & (D_M > 1e-6) & jnp.isfinite(R_M_val) & jnp.isfinite(D_M)
    capacity = jax.lax.cond(is_valid, compute_capacity, return_nan, R_M_val, D_M)

    # --- 4. Alignment Metrics ---
    norms_0 = jnp.linalg.norm(s_0, axis=-1, keepdims=True)
    s_0_normed = s_0 / (norms_0 + 1e-9)
    cos_sim_0 = jnp.abs(s_0_normed @ s_0_normed.T)
    mask = 1.0 - jnp.eye(P)
    rho_c = jnp.sum(cos_sim_0 * mask) / (P * (P - 1))
    
    def compute_axis_corr(s1_k):
        norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
        s1_normed = s1_k / (norms_1 + 1e-9)
        cos_sim_1 = jnp.abs(s1_normed @ s1_normed.T)
        return jnp.sum(cos_sim_1 * mask) / (P * (P - 1))
    
    rho_a = jnp.mean(jax.vmap(compute_axis_corr)(s_1))
    
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

# --- Simulated Capacity Kernels (Algorithm 1) ---

# indices: 0:key, 1:flat_manifolds, 2:n_proj, 3:M_per_manifold, 4:P
def check_linear_separability_batch(key, flat_manifolds, n_proj, M_per_manifold, P):
    """
    Checks if randomly projected manifolds are linearly separable.
    """
    N_ambient = flat_manifolds.shape[1]
    
    # 1. Random Projection: Pi from R^N -> R^n
    k1, k2 = jax.random.split(key)
    Pi = jax.random.normal(k1, (N_ambient, n_proj)) / jnp.sqrt(n_proj)
    
    # Project data
    projected_data = flat_manifolds @ Pi
    
    # 2. Random Dichotomy
    y = jax.random.choice(k2, jnp.array([-1.0, 1.0]), (P,))
    y_expanded = jnp.repeat(y, M_per_manifold) # (P*M,)
    
    # 3. Solve Linear Separability (SVM-like feasibility)
    G_mat = -(y_expanded[:, None] * projected_data) # (P*M, n)
    h_vec = -jnp.ones((P * M_per_manifold,))
    
    Q_mat = jnp.eye(n_proj)
    q_vec = jnp.zeros((n_proj,))
    
    solver = OSQP(tol=1e-3, maxiter=1000, verbose=False, check_primal_dual_infeasability=True)
    sol = solver.run(params_obj=(Q_mat, q_vec), params_eq=None, params_ineq=(G_mat, h_vec))
    
    # Verify margin manually
    w_opt = sol.params.primal
    margins = y_expanded * (projected_data @ w_opt)
    
    # If minimum margin is >= 1 - epsilon, it is separable
    is_separable = jnp.min(margins) >= 0.99 
    return is_separable

def estimate_separability_probability(key, flat_manifolds, n_proj, M: int, P: int, m_trials=100):
    """Estimates p_n: Probability that manifolds are separable in n dimensions."""
    keys = jax.random.split(key, m_trials)
    check_fn = lambda k: check_linear_separability_batch(k, flat_manifolds, n_proj, M, P)
    results = jax.vmap(check_fn)(keys)
    return jnp.mean(results.astype(jnp.float32))

def compute_simulated_capacity(rng, representations, labels):
    """
    Implements the Binary Search algorithm (Algorithm 1) to find Simulated Capacity.
    Fully uses JAX for random keys and array manipulation.
    """
    # Eager execution for data organization (dynamic shapes)
    unique_labels = jnp.unique(labels)
    P = unique_labels.shape[0]
    
    if P < 2: return jnp.nan
    
    # Organize data (ensure equal points M per manifold)
    counts = jnp.array([jnp.sum(labels == l) for l in unique_labels])
    M = int(jnp.min(counts))
    
    if M < 2: return jnp.nan
    
    grouped_data = []
    # Loop over classes in standard python, but operations are JAX
    for i in range(P):
        l = unique_labels[i]
        # Boolean masking returns dynamic shape, handled by JAX eager execution
        idxs = jnp.where(labels == l)[0][:M]
        grouped_data.append(representations[idxs])
    
    manifolds = jnp.stack(grouped_data) # (P, M, N)
    flat_manifolds = manifolds.reshape(-1, manifolds.shape[-1])
    N_ambient = flat_manifolds.shape[1]

    # Binary Search for n*
    # We want smallest n such that p_n >= 0.5
    n_left = 1
    n_right = N_ambient
    n_star = N_ambient
    
    iteration = 0
    while n_left <= n_right:
        n_mid = (n_left + n_right) // 2
        if n_mid == 0: n_mid = 1
        
        # Split key for this iteration's trials
        rng, iter_key = jax.random.split(rng)
        
        p_n = estimate_separability_probability(
            iter_key, 
            flat_manifolds, 
            int(n_mid), 
            int(M), 
            int(P), 
            m_trials=100
        )
        
        if p_n >= 0.5:
            n_star = n_mid
            n_right = n_mid - 1 
        else:
            n_left = n_mid + 1 
            
        iteration += 1
        
    alpha_sim = P / n_star
    return float(alpha_sim)


def run_manifold_geometry(rng, representations, labels, n_samples_t=50):
    """
    Main entry point for computing manifold metrics (GLUE).
    """
    # Check for NaNs
    if not jnp.all(jnp.isfinite(representations)):
        print("Warning: NaN or Inf detected in representations")
        return {k: jnp.nan for k in ['Capacity', 'Radius', 'Dimension', 
                                   'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']}
    
    unique_labels = jnp.unique(labels)
    P = unique_labels.shape[0]
    N = representations.shape[1]
    
    counts = jnp.array([jnp.sum(labels == l) for l in unique_labels])
    M = int(jnp.min(counts))
    
    if M < 2:
        return {k: jnp.nan for k in ['Capacity', 'Radius', 'Dimension', 
                                   'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']}

    grouped_data = []
    for i in range(P):
        l = unique_labels[i]
        idxs = jnp.where(labels == l)[0][:M]
        grouped_data.append(representations[idxs])
    
    manifolds = jnp.stack(grouped_data)
    flat_manifolds = manifolds.reshape(-1, N)
    
    # Generate keys for the solver
    keys = jax.random.split(rng, n_samples_t)
    solve_batch = jax.vmap(solve_single_anchor, in_axes=(0, None, None, None, None))
    
    # Run Solver
    s_raw, t_vecs = solve_batch(keys, flat_manifolds, M, P, N)
    metrics_jax = compute_metrics_from_anchors(s_raw, t_vecs)
    
    return {k: float(v) for k, v in metrics_jax.items()}

# --- Analysis Pipeline Integration ---

def analyze_manifold_trajectory(config, task_names):
    """
    Loads saved representations and computes manifold metrics over time.
    Uses JAX for all computations and key management.
    """
    print("\n--- Starting Manifold Geometric Analysis (GLUE) [JAX Backend] ---")
    
    # Initialize the master key from config seed
    master_key = jax.random.PRNGKey(config.seed)

    glue_metrics = ['Capacity', 'Radius', 'Dimension', 
                    'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']
    
    extra_metrics = ['Simulated_Capacity', 'Capacity_Relative_Error']
    all_stored_metrics = glue_metrics + extra_metrics

    # Dictionary to store full results per task
    full_results = {}
    
    # For plotting aggregate history (only GLUE metrics)
    plot_history = {k: [] for k in glue_metrics}
    
    # Store Simulated Capacity Results for Overlay
    sim_cap_history = {}
    
    task_boundaries = []
    current_epoch = 0
    
    for t_idx, t_name in enumerate(task_names):
        rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
        lbl_path = os.path.join(config.reps_dir, f"{t_name}_labels.npy")
        
        if not os.path.exists(rep_path) or not os.path.exists(lbl_path):
            print(f"Skipping {t_name} (files not found)")
            continue
            
        # Load using JAX (via numpy then convert)
        with open(rep_path, 'rb') as f:
            reps_data = jnp.load(f)
        with open(lbl_path, 'rb') as f:
            labels = jnp.load(f)    
        
        # Expecting reps_data: (Steps, Repeats, Samples, Dim)
        n_steps, n_repeats, n_samples, dim = reps_data.shape
        
        # Asserts for system integrity
        assert n_repeats == config.n_repeats, f"Config repeats {config.n_repeats} mismatch data {n_repeats}"
        
        # Handle Labels: Expect (Samples, Repeats) from single_run
        if labels.ndim == 2:
            assert labels.shape == (n_samples, n_repeats), f"Label shape {labels.shape} mismatch (N, P)"
            labels_reshaped = labels
        else:
            # Fallback if flattened
            assert labels.size == n_samples * n_repeats
            labels_reshaped = labels.reshape(n_samples, n_repeats)
        
        print(f"Processing {t_name}: {n_steps} steps, {n_repeats} repeats...")
        
        # Storage for current task
        task_metrics_lists = {k: [] for k in all_stored_metrics}
        
        # Temp storage for Sim Cap overlay
        task_sim_epochs = []
        task_sim_vals = [] 
        
        for step in range(n_steps):
            epoch_num = current_epoch + (step + 1) * config.log_frequency
            
            # Temporary list for this step's repeats
            step_res = {k: [] for k in all_stored_metrics}
            
            # Run Simulated Capacity every 100 epochs
            run_sim_cap = (epoch_num % 100 == 0)
            
            for r in range(n_repeats):
                # Fold key for specific Step and Repeat to ensure uniqueness and reproducibility
                # Key structure: Master -> Task -> Step -> Repeat
                step_key = jax.random.fold_in(master_key, t_idx)
                step_key = jax.random.fold_in(step_key, step)
                unique_key = jax.random.fold_in(step_key, r)
                
                # Split key for separate sub-routines (GLUE vs SimCap)
                glue_key, sim_key = jax.random.split(unique_key)
                
                curr_reps = reps_data[step, r]
                curr_labels = labels_reshaped[:, r]
                
                # Check for NaNs in data
                if not jnp.all(jnp.isfinite(curr_reps)):
                    for k in all_stored_metrics: step_res[k].append(jnp.nan)
                    continue

                # Subsample data to keep QP solver efficient
                # We select 'config.analysis_subsamples' per class (using the config variable)
                unique_classes = jnp.unique(curr_labels)
                indices_list = []
                for cls in unique_classes:
                    # Get all indices for this class
                    cls_idxs = jnp.where(curr_labels == cls)[0]
                    
                    # Randomly shuffle and pick top K
                    # Use a subkey for permutation to ensure randomness
                    perm_key = jax.random.fold_in(unique_key, int(cls))
                    shuffled_cls_idxs = jax.random.permutation(perm_key, cls_idxs)
                    
                    # Update: Use analysis_subsamples from config
                    limit = min(len(cls_idxs), config.analysis_subsamples)
                    indices_list.append(shuffled_cls_idxs[:limit])
                
                # Concatenate and filter data
                selected_idxs = jnp.concatenate(indices_list)
                curr_reps_sub = curr_reps[selected_idxs]
                curr_labels_sub = curr_labels[selected_idxs]

                try:
                    # 1. Standard GLUE Metrics
                    res = run_manifold_geometry(glue_key, curr_reps_sub, curr_labels_sub, n_samples_t=config.n_t)
                    for k in glue_metrics: step_res[k].append(res[k])
                    
                    # 2. Simulated Capacity & Accuracy
                    if run_sim_cap:
                        sc = compute_simulated_capacity(sim_key, curr_reps, curr_labels)
                        glue_cap = res['Capacity']
                        
                        # Calculate Relative Error
                        if glue_cap > 1e-9:
                            rel_error = abs(sc - glue_cap) / glue_cap
                        else:
                            rel_error = jnp.nan
                            
                        step_res['Simulated_Capacity'].append(sc)
                        step_res['Capacity_Relative_Error'].append(rel_error)
                    else:
                        step_res['Simulated_Capacity'].append(jnp.nan)
                        step_res['Capacity_Relative_Error'].append(jnp.nan)

                except Exception as e:
                    print(f"Error at step {step} rep {r}: {e}")
                    for k in all_stored_metrics: step_res[k].append(jnp.nan)

            # Store aggregated step data
            for k in all_stored_metrics:
                task_metrics_lists[k].append(step_res[k])
                
            # Logging & Overlay storage
            if run_sim_cap:
                sim_caps = jnp.array(step_res['Simulated_Capacity'])
                
                # Note: We use jnp.nanmean
                task_sim_epochs.append(epoch_num)
                task_sim_vals.append(sim_caps)

        # Finalize Task Data
        full_results[t_name] = {}
        for k in all_stored_metrics:
            full_results[t_name][k] = jnp.array(task_metrics_lists[k])
            
        # Add to plotting history (only GLUE metrics)
        for k in glue_metrics:
            plot_history[k].extend(task_metrics_lists[k])
            
        # Process Sim Cap for overlay
        if task_sim_epochs:
            sim_vals_arr = jnp.array(task_sim_vals)
            sim_means = jnp.nanmean(sim_vals_arr, axis=1)
            sim_stds = jnp.nanstd(sim_vals_arr, axis=1)
            sim_cap_history[t_name] = (task_sim_epochs, sim_means, sim_stds)

        current_epoch += n_steps * config.log_frequency
        task_boundaries.append(current_epoch)

    # Convert plot history to simple arrays for matplotlib
    # Note: We must convert JAX arrays to Numpy for Matplotlib
    for k in plot_history:
        if len(plot_history[k]) > 0:
            plot_history[k] = jnp.array(plot_history[k]) 

    # --- Plotting ---
    if len(plot_history['Capacity']) == 0:
        print("No manifold metrics computed.")
        return full_results

    n_metrics = len(glue_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: axes = [axes]
    
    total_steps = len(plot_history[glue_metrics[0]])
    x_axis = jnp.arange(1, total_steps + 1) * config.log_frequency
    
    colors = plt.cm.viridis(jnp.linspace(0, 0.9, n_metrics))
    
    for i, metric in enumerate(glue_metrics):
        ax = axes[i]
        data = plot_history[metric]
        
        if data.size == 0: continue
            
        mean = jnp.nanmean(data, axis=1)
        std = jnp.nanstd(data, axis=1)
        
        color = colors[i]
        ax.plot(x_axis, mean, label=f"GLUE {metric}", color=color, linewidth=2)
        ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)
        
        # --- Overlay Simulated Capacity ---
        if metric == 'Capacity':
            for t_name, (epochs, means, stds) in sim_cap_history.items():
                ax.errorbar(epochs, means, yerr=stds, fmt='o', color='black', 
                            ecolor='gray', elinewidth=2, capsize=4, markersize=5,
                            label='Simulated (Alg 1)' if t_name == task_names[0] else None)
            ax.legend()
        
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        for boundary in task_boundaries[:-1]:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
        if i == 0:
            ax.set_title(f"Manifold Geometric Analysis - {config.dataset_name}")

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    save_path = os.path.join(config.figures_dir, f"manifold_metrics_{config.dataset_name}.png")
    plt.savefig(save_path)
    print(f"Manifold analysis plots saved to {save_path}")
    
    return full_results