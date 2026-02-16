import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70" 

import jax
import gc
import jax.numpy as jnp
import matplotlib.pyplot as plt
from glue_module.glue_analysis import run_glue_solver
import numpy as np
from jaxopt import OSQP
from functools import partial

print(f"Using {jax.devices()}")

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# ----------------------------------------------------------------------
# Helper functions (converted to JAX)
# ----------------------------------------------------------------------
def get_input_means_jax(key, N, target_dot_product_matrix):
    """
    JAX version of get_input_means.
    Generates P unit vectors in N-dim space with specified pairwise dot products.
    """
    P = target_dot_product_matrix.shape[0]
    # 1. Factorize the target matrix
    try:
        L = jnp.linalg.cholesky(target_dot_product_matrix)
    except:
        # fallback to eigendecomposition
        evals, evecs = jnp.linalg.eigh(target_dot_product_matrix)
        evals = jnp.maximum(evals, 0.0)
        L = evecs @ jnp.diag(jnp.sqrt(evals))
    X = L.T  # (P, P)

    # 2. Generate an orthonormal basis in R^N (N x P)
    key, subkey = jax.random.split(key)
    E, _ = jnp.linalg.qr(jax.random.normal(subkey, (N, P)))

    # 3. Construct final vectors
    V = E @ X  # (N, P)
    return V.T, key  # (P, N)


def get_input_subspaces_jax(key, N, D, P, target_sam_matrix):
    """
    JAX version of get_input_subspaces.
    Generates P sets of D orthonormal vectors with specified pairwise alignments.
    Returns a list of (D, N) arrays.
    """
    # Validate dimensions
    if N < P * D:
        raise ValueError(f"N must be >= P*D, got N={N}, P*D={P*D}")

    # Construct the Gram matrix from SAM: G_ij = sqrt(target_sam_ij / D)
    gram_matrix = jnp.sqrt(target_sam_matrix / D)

    # Factorize the Gram matrix
    evals, evecs = jnp.linalg.eigh(gram_matrix)
    evals = jnp.maximum(evals, 0.0)
    sqrt_lambda = jnp.diag(jnp.sqrt(evals))
    C = evecs @ sqrt_lambda  # (P, P) mixing matrix

    # Generate a pool of P*D orthonormal vectors in R^N
    key, subkey = jax.random.split(key)
    master_pool, _ = jnp.linalg.qr(jax.random.normal(subkey, (N, P * D)))

    generated_sets = []
    for i in range(P):
        U_i = jnp.zeros((N, D))
        for k in range(P):
            V_k = master_pool[:, k*D:(k+1)*D]          # (N, D)
            U_i += C[i, k] * V_k
        generated_sets.append(U_i.T)  # (D, N)

    return generated_sets, key


# ----------------------------------------------------------------------
# Updated ground truth data generator
# ----------------------------------------------------------------------
@partial(jax.jit, static_argnames=('P', 'N', 'n_points', 'D'))
def generate_glue_ground_truth_data(key, P, N, n_points, R, D, rho_c, rho_a, psi):
    """
    Generate P point clouds of N-dimensional points with controlled geometry.
    - Centers are unit vectors with pairwise dot product rho_c.
    - Axes are orthonormal bases with subspace alignment determined by rho_a.
    - Point generation follows the original GLUE pipeline.
    """
    # --- Construct target correlation matrices ---
    # Centers: compound symmetric with correlation rho_c
    # Ensure the matrix is positive semi-definite: rho_c ∈ [0,1] is safe.
    target_center_dot = (1.0 - rho_c) * jnp.eye(P) + rho_c * jnp.ones((P, P))

    # Axes: target SAM matrix.
    # For orthonormal axes, SAM_ii = D (by definition).
    # SAM_ij = D^2 * rho_a^2  (if each pair of axes had correlation rho_a)
    # Clamp to D to keep matrix PSD (rho_a must be ≤ 1/sqrt(D)).
    sam_offdiag = jnp.minimum(D**2 * rho_a**2, D)
    target_sam = jnp.eye(P) * D + (1.0 - jnp.eye(P)) * sam_offdiag

    # --- Generate centers ---
    key, subkey = jax.random.split(key)
    centers, key = get_input_means_jax(subkey, N, target_center_dot)   # (P, N)

    # --- Generate axes (orthonormal bases) ---
    key, subkey = jax.random.split(key)
    axes_list, key = get_input_subspaces_jax(subkey, N, D, P, target_sam)
    bases = jnp.stack(axes_list, axis=0)  # (P, D, N)

    # --- Generate points (same as original) ---
    scale = N ** (-0.5)

    # Random longitudinal modulation (a_j)
    key, subkey = jax.random.split(key)
    a_j = jax.random.normal(subkey, (P, n_points, 1)) * psi   # (P, n_points, 1)

    # Random directions in the D-dimensional latent space (b_j), normalized
    key, subkey = jax.random.split(key)
    raw_b = jax.random.normal(subkey, (P, n_points, D))
    # norms = jnp.linalg.norm(raw_b, axis=2, keepdims=True)
    # b_j = raw_b / (norms + 1e-9)

    # Intrinsic variation = b_j @ bases, then normalized
    intrinsic_variation = jnp.matmul(raw_b, bases)
    # intrinsic_variation = jnp.matmul(b_j, bases)               # (P, n_points, N)
    # iv_norms = jnp.linalg.norm(intrinsic_variation, axis=2, keepdims=True)
    # intrinsic_variation = intrinsic_variation / (iv_norms + 1e-9)

    # Combine
    longitudinal_part = (1 + a_j) * centers[:, None, :]        # (P, n_points, N)
    points = longitudinal_part + (R * intrinsic_variation)

    return points, centers

def main():
    # --- Parameters ---
    P = 2
    N = 50
    M = 100
    n_points = M
    rho_c = 0.0
    rho_a = 0.0
    psi = 0.0
    n_t = 100
    n_reps = 20
    qp = OSQP(tol=1e-4)

    radii_scan = jnp.linspace(0.2, 2.0, 10)
    dims_scan = [2, 4, 6, 8, 10]

    main_key = jax.random.PRNGKey(0)

    # --- Data Generation Phase ---

    @partial(jax.jit, static_argnames=['D', 'n_reps'])
    def generate_batch_data(key, radii_array, D, n_reps):
        n_radii = radii_array.shape[0]
        
        keys = jax.random.split(key, n_radii * n_reps)
        keys = keys.reshape(n_radii, n_reps, -1)
        
        def generate_single(k, r):
            d, _ = generate_glue_ground_truth_data(
                k, P, N, n_points, r, D, rho_c, rho_a, psi
            )
            return d

        vmapped_gen = jax.vmap(generate_single, in_axes=(0, None))
        
        def scan_gen(keys_batch, r):
            return vmapped_gen(keys_batch, r)

        batch_data = jax.lax.map(lambda args: scan_gen(*args), (keys, radii_array))
        return batch_data

    # --- Solver Phase ---

    @jax.jit
    def solve_batch_scan(key, batch_data):
        n_radii = batch_data.shape[0]
        n_reps = batch_data.shape[1]
        
        keys = jax.random.split(key, n_radii * n_reps)
        keys = keys.reshape(n_radii, n_reps, -1)

        def run_single_solve(k, d):
            metrics, _ = run_glue_solver(k, d, P, M, N, n_t, qp)
            
            return jnp.stack([
                metrics[0], # Capacity
                metrics[1], # Dimension
                metrics[2], # Radius
                metrics[6], # Approx Capacity
                metrics[3], # Center Align
                metrics[4], # Axis Align
                metrics[5]  # Center-Axis Align
            ])

        def scan_body(carry, x):
            batch_keys, batch_data_slice = x
            batch_results = jax.vmap(run_single_solve)(batch_keys, batch_data_slice)
            
            means = jnp.mean(batch_results, axis=0)
            stds = jnp.std(batch_results, axis=0)
            
            stats = jnp.ravel(jnp.stack([means, stds], axis=1))
            return carry, stats

        _, results_stack = jax.lax.scan(scan_body, None, (keys, batch_data))
        return results_stack

    # --- Main Execution Loop ---

    print(f"--- Starting Decoupled Scan ---")
    results_collection = []

    for i, D in enumerate(dims_scan):
        print(f"Processing Dimension D={D}...", flush=True)
        
        main_key, gen_key, solve_key = jax.random.split(main_key, 3)
        
        data_batch = generate_batch_data(gen_key, radii_scan, D, n_reps)
        data_batch.block_until_ready()
        
        print(f"  > Running Solver...", flush=True)
        dim_results = solve_batch_scan(solve_key, data_batch)
        dim_results.block_until_ready()
        
        results_collection.append(dim_results)

        del data_batch
        gc.collect()

    # --- Post-Processing ---

    all_results_tensor = jnp.stack(results_collection)
    
    radii_gt = radii_scan 
    dims_gt = jnp.array(dims_scan)

    # (Name, mean_idx, std_idx)
    metrics_map = [
        ("Capacity", 0, 1),
        ("Capacity Approx", 6, 7),
        ("Calculated Dimension", 2, 3),
        ("Calculated Radius", 4, 5),
        ("Center Alignment", 8, 9),
        ("Axis Alignment", 10, 11),
        ("Center-Axis Alignment", 12, 13)
    ]

    metrics_data = []
    for name, m_idx, s_idx in metrics_map:
        metrics_data.append((
            name,
            all_results_tensor[:, :, m_idx],
            all_results_tensor[:, :, s_idx]
        ))

    # --- Plotting ---

    fig, axes = plt.subplots(2, 7, figsize=(35, 10), constrained_layout=True)

    cm_dims = plt.get_cmap('viridis')(np.linspace(0, 1, len(dims_gt)))
    cm_radii = plt.get_cmap('plasma')(np.linspace(0, 1, len(radii_gt)))

    # Row 1: X-axis = Radius, Fixed Dimension Lines
    for col_idx, (name, mean_grid, std_grid) in enumerate(metrics_data):
        ax = axes[0, col_idx]
        
        for i, D in enumerate(dims_gt):
            y_mean = mean_grid[i, :]
            y_std = std_grid[i, :]
            
            ax.plot(radii_gt, y_mean, marker='o', markersize=3, 
                    label=f'D={D}', color=cm_dims[i])
            
            ax.fill_between(radii_gt, y_mean - y_std, y_mean + y_std, 
                            color=cm_dims[i], alpha=0.1)

        if name == "Calculated Radius":
            ax.plot(radii_gt, radii_gt, 'k--', alpha=0.5, linewidth=2, label='Identity (GT)')
        
        ax.set_title(f"{name} vs Radius")
        ax.set_xlabel("Ground Truth Radius")
        ax.set_ylabel(name)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if col_idx == 0:
            ax.legend(title="Fixed Dimension", fontsize='small')

    # Row 2: X-axis = Dimension, Fixed Radius Lines
    for col_idx, (name, mean_grid, std_grid) in enumerate(metrics_data):
        ax = axes[1, col_idx]
        
        for j, R in enumerate(radii_gt):
            y_mean = mean_grid[:, j]
            y_std = std_grid[:, j]
            
            ax.plot(dims_gt, y_mean, marker='s', markersize=3, 
                    label=f'R={R:.1f}', color=cm_radii[j])
            
            ax.fill_between(dims_gt, y_mean - y_std, y_mean + y_std, 
                            color=cm_radii[j], alpha=0.05)

        if name == "Calculated Dimension":
            ax.plot(dims_gt, dims_gt, 'k--', alpha=0.5, linewidth=2, label='Identity (GT)')

        ax.set_title(f"{name} vs Dimension")
        ax.set_xlabel("Ground Truth Dimension")
        ax.set_ylabel(name)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xticks(dims_gt)
        
        if col_idx == 0:
            ax.legend(title="Fixed Radius", fontsize='small', ncol=2)

    fig.suptitle(f"GLUE Sweep Results ({n_reps} reps)", fontsize=16)
    plt.savefig("./figS6.png")
    print(f"Finshed. Saved Plots.")

if __name__=='__main__':
    main()