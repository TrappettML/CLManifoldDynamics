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

@partial(jax.jit, static_argnames=('P', 'N', 'n_points', 'D')) 
def generate_glue_ground_truth_data(key, P, N, n_points, R, D, rho_c, rho_a, psi):
    """
    Optimized generation of P point clouds with dynamic manifold dimension D.
    """
    k_c, k_b, k_pts = jax.random.split(key, 3)
    scale = N**(-0.5)
    
    # Analytic Mixing Coefficients
    c_unique_c = jnp.sqrt(jnp.maximum(1.0 - rho_c**2, 1e-10))
    c_common_c = rho_c
    
    c_unique_a = jnp.sqrt(jnp.maximum(1.0 - rho_a**2, 1e-10))
    c_common_a = rho_a

    # Generate Correlated Centers (u_0)
    k_c_u, k_c_s = jax.random.split(k_c)
    U_c_unique = jax.random.normal(k_c_u, (P, N)) * scale
    U_c_shared = jax.random.normal(k_c_s, (1, N)) * scale
    
    centers = c_unique_c * U_c_unique + c_common_c * U_c_shared

    # Bases
    k_b_u, k_b_s = jax.random.split(k_b)
    U_b_unique = jax.random.normal(k_b_u, (D, P, N)) * scale
    U_b_shared = jax.random.normal(k_b_s, (D, 1, N)) * scale
    
    bases = c_unique_a * U_b_unique + c_common_a * U_b_shared
    bases = jnp.swapaxes(bases, 0, 1) # (P, D, N)

    # Noise and intrinsic variation
    k_pts_a, k_pts_b = jax.random.split(k_pts)
    a_j = jax.random.normal(k_pts_a, (P, n_points, 1)) * psi
    
    raw_b = jax.random.normal(k_pts_b, (P, n_points, D))
    norms = jnp.linalg.norm(raw_b, axis=2, keepdims=True)
    b_j = raw_b / (norms + 1e-9)
    
    intrinsic_variation = jnp.matmul(b_j, bases)
    iv_norms = jnp.linalg.norm(intrinsic_variation, axis=2, keepdims=True)
    intrinsic_variation = intrinsic_variation / (iv_norms + 1e-9)
    
    # Combine
    longitudinal_part = (1 + a_j) * centers[:, None, :]
    points = longitudinal_part + (R * intrinsic_variation)
    
    return points, centers

def main():
    # --- Parameters ---
    P = 2
    N = 40
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
        ("Calculated Dimension", 2, 3),
        ("Calculated Radius", 4, 5),
        ("Capacity Approx", 6, 7),
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

if __name__=='__main__':
    main()