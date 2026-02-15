import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70" # set a hard cap on GPU mem

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
    
    Args:
        max_D: A static upper bound for D. Allocates arrays of this size 
               to avoid recompilation when D changes.
    """
    # Split keys
    k_c, k_b, k_pts = jax.random.split(key, 3)
    
    # Scale factor
    scale = N**(-0.5)
    
    # --- Analytic Mixing Coefficients ---
    # Clip for numerical stability
    c_unique_c = jnp.sqrt(jnp.maximum(1.0 - rho_c**2, 1e-10))
    c_common_c = rho_c
    
    c_unique_a = jnp.sqrt(jnp.maximum(1.0 - rho_a**2, 1e-10))
    c_common_a = rho_a

    # --- Step 1 & 2: Generate Correlated Centers (u_0) ---
    # Shape: (P, N) - Independent of D, so logic remains identical
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

    # Noise
    k_pts_a, k_pts_b = jax.random.split(k_pts)
    a_j = jax.random.normal(k_pts_a, (P, n_points, 1)) * psi
    
    # Sample directly in R^D
    raw_b = jax.random.normal(k_pts_b, (P, n_points, D))
    # norms = jnp.linalg.norm(raw_b, axis=2, keepdims=True)
    b_j = raw_b #/ (norms + 1e-9)
    
    # Matrix Multiplication
    # This is now lightning fast for small D
    intrinsic_variation = jnp.matmul(b_j, bases)
    
    # 4. Combine
    # centers: (P, N) -> (P, 1, N)
    longitudinal_part = (1 + a_j) * centers[:, None, :]
    
    # Result shape is (P, n_points, N), which is static given P, N, n_points
    points = longitudinal_part + (R * intrinsic_variation)
    
    return points, centers

def main():
    # --- 1. Setup Parameters ---
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

    # Scan Ranges (Mesh Grid)
    radii_scan = jnp.linspace(0.2, 2.0, 10)
    dims_scan = [2, 4, 6, 8, 10]

    main_key = jax.random.PRNGKey(0)

    # --- 2. Phase 1: Data Generation ---

    @partial(jax.jit, static_argnames=['D', 'n_reps'])
    def generate_batch_data(key, radii_array, D, n_reps):
        """
        Generates ALL data for a specific Dimension D.
        Output Shape: (n_radii, n_reps, [Data Shape...])
        """
        n_radii = radii_array.shape[0]
        
        # 1. Create a grid of keys: (n_radii, n_reps)
        keys = jax.random.split(key, n_radii * n_reps)
        keys = keys.reshape(n_radii, n_reps, -1)
        
        # 2. Define single generation function
        def generate_single(k, r):
            # We only keep 'data' here for the solver, it returns data, centers
            d, _ = generate_glue_ground_truth_data(
                k, P, N, n_points, r, D, rho_c, rho_a, psi
            )
            return d

        # 3. Vectorize over Reps (inner), Map over Radii (outer)
        # vmap over keys (axis 0), broadcast radius
        vmapped_gen = jax.vmap(generate_single, in_axes=(0, None))
        
        # scan/map over radii
        # We map over (keys[i], radii_array[i])
        def scan_gen(keys_batch, r):
            return vmapped_gen(keys_batch, r)

        batch_data = jax.lax.map(lambda args: scan_gen(*args), (keys, radii_array))
        
        return batch_data

    # --- 3. Phase 2: Solver Execution ---
    @jax.jit
    def solve_batch_scan(key, batch_data):
        """
        Solves a pre-generated batch of data.
        Input Data Shape: (n_radii, n_reps, ...)
        """
        n_radii = batch_data.shape[0]
        n_reps = batch_data.shape[1]
        
        # Prepare keys for the solver phase
        # One key per (radius, rep)
        keys = jax.random.split(key, n_radii * n_reps)
        keys = keys.reshape(n_radii, n_reps, -1)

        # Define single solver run
        def run_single_solve(k, d):
            metrics, _ = run_glue_solver(k, d, P, M, N, n_t, qp)
            # Extract metrics: [Capacity, Theory_D, Theory_R, CapApprox]
            return jnp.stack([metrics[0], metrics[1], metrics[2], metrics[3]])

        # Define the scanning function (over Radii)
        def scan_body(carry, x):
            # x contains: (keys_for_this_radius_batch, data_for_this_radius_batch)
            batch_keys, batch_data_slice = x
            
            # Vmap over the repetitions
            batch_results = jax.vmap(run_single_solve)(batch_keys, batch_data_slice)
            
            # Calculate stats for this radius (mean/std over the reps axis)
            # batch_results shape: (n_reps, 4)
            means = jnp.mean(batch_results, axis=0) # Shape (4,)
            stds = jnp.std(batch_results, axis=0)   # Shape (4,)
            
            # Interleave mean/std: [Mean_Cap, Std_Cap, Mean_D, Std_D, ...]
            # Stack them to get shape (8,)
            stats = jnp.ravel(jnp.stack([means, stds], axis=1))
            
            return carry, stats # Carry is unused but required for scan

        # Run the scan
        # We scan over the leading axis of keys and batch_data
        _, results_stack = jax.lax.scan(scan_body, None, (keys, batch_data))
        
        return results_stack

    # --- 4. Main Execution Loop ---

    print(f"--- Starting Decoupled Scan ---")
    results_collection = []

    for i, D in enumerate(dims_scan):
        print(f"Processing Dimension D={D}...", flush=True)
        
        # Split keys for this dimension
        main_key, gen_key, solve_key = jax.random.split(main_key, 3)
        
        # --- PHASE 1: GENERATE ---
        # Result: (n_radii, n_reps, Data_Dim...)
        data_batch = generate_batch_data(gen_key, radii_scan, D, n_reps)
        
        # Optional: Force computation to ensure memory is actually allocated/managed
        data_batch.block_until_ready()
        
        # --- PHASE 2: SOLVE ---
        # Result: (n_radii, 8)
        print(f"  > Running Solver...", flush=True)
        dim_results = solve_batch_scan(solve_key, data_batch)
        dim_results.block_until_ready()
        
        results_collection.append(dim_results)

        del data_batch
        gc.collect()


    # --- 5. Post-Processing ---

    all_results_tensor = jnp.stack(results_collection)

    results_grid = {
        'scan_dims': jnp.array(dims_scan),
        'scan_radii': radii_scan,
        'grid_means_Cap': all_results_tensor[:, :, 0],
        'grid_stds_Cap':  all_results_tensor[:, :, 1],
        'grid_means_D':   all_results_tensor[:, :, 2],
        'grid_stds_D':    all_results_tensor[:, :, 3],
        'grid_means_R':   all_results_tensor[:, :, 4],
        'grid_stds_R':    all_results_tensor[:, :, 5],
        'grid_means_CapApprox': all_results_tensor[:, :, 6],
        'grid_stds_CapApprox':  all_results_tensor[:, :, 7],
    }

    # --- Data Preparation ---
    # Unpacking data from the results dictionary for easier access
    radii_gt = results_grid['scan_radii']  # X-axis for Row 1
    dims_gt = results_grid['scan_dims']    # X-axis for Row 2

    # Organize metrics into a list of tuples: (Name, Mean_Array, Std_Array)
    # Arrays are shape (n_dims, n_radii)
    metrics_data = [
        ("Capacity", results_grid['grid_means_Cap'], results_grid['grid_stds_Cap']),
        ("Capacity Approx", results_grid['grid_means_CapApprox'], results_grid['grid_stds_CapApprox']),
        ("Calculated Radius", results_grid['grid_means_R'], results_grid['grid_stds_R']),
        ("Calculated Dimension", results_grid['grid_means_D'], results_grid['grid_stds_D'])
    ]

    # --- Plotting Setup ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)

    # Color maps for distinct lines
    cm_dims = plt.get_cmap('viridis')(np.linspace(0, 1, len(dims_gt)))
    cm_radii = plt.get_cmap('plasma')(np.linspace(0, 1, len(radii_gt)))

    # ==========================================
    # ROW 1: X-axis = Ground Truth Radius
    # Legend = Dimensions (Lines are fixed D)
    # ==========================================
    for col_idx, (name, mean_grid, std_grid) in enumerate(metrics_data):
        ax = axes[0, col_idx]
        
        # Iterate over Dimensions (rows of the grid)
        for i, D in enumerate(dims_gt):
            # Slice: Fix D (row i), vary R (all columns)
            y_mean = mean_grid[i, :]
            y_std = std_grid[i, :]
            
            ax.plot(radii_gt, y_mean, marker='o', markersize=3, 
                    label=f'D={D}', color=cm_dims[i])
            
            # Optional: Shading for Std Dev
            ax.fill_between(radii_gt, y_mean - y_std, y_mean + y_std, 
                            color=cm_dims[i], alpha=0.1)

        # Reference Lines for Columns 3 and 4 (Radius and Dim)
        if col_idx == 2: # Calculated Radius vs GT Radius
            ax.plot(radii_gt, radii_gt, 'k--', alpha=0.5, linewidth=2, label='Identity (GT)')
        if col_idx == 3: # Calculated Dim vs GT Radius (Expected constant D lines)
            pass 

        ax.set_title(f"{name} vs Radius")
        ax.set_xlabel("Ground Truth Radius")
        ax.set_ylabel(name)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Only add legend to the first plot to avoid clutter, or add outside
        if col_idx == 0:
            ax.legend(title="Fixed Dimension", fontsize='small')

    # ==========================================
    # ROW 2: X-axis = Ground Truth Dimension
    # Legend = Radii (Lines are fixed R)
    # ==========================================
    for col_idx, (name, mean_grid, std_grid) in enumerate(metrics_data):
        ax = axes[1, col_idx]
        
        # Iterate over Radii (columns of the grid)
        for j, R in enumerate(radii_gt):
            # Slice: Fix R (col j), vary D (all rows)
            y_mean = mean_grid[:, j]
            y_std = std_grid[:, j]
            
            ax.plot(dims_gt, y_mean, marker='s', markersize=3, 
                    label=f'R={R:.1f}', color=cm_radii[j])
            
            # Optional: Shading
            ax.fill_between(dims_gt, y_mean - y_std, y_mean + y_std, 
                            color=cm_radii[j], alpha=0.05)

        # Reference Lines
        if col_idx == 3: # Calculated Dim vs GT Dim
            ax.plot(dims_gt, dims_gt, 'k--', alpha=0.5, linewidth=2, label='Identity (GT)')

        ax.set_title(f"{name} vs Dimension")
        ax.set_xlabel("Ground Truth Dimension")
        ax.set_ylabel(name)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Handle X-ticks for Dimensions (integers)
        ax.set_xticks(dims_gt)
        
        # Only add legend to the first plot of the row
        if col_idx == 0:
            ax.legend(title="Fixed Radius", fontsize='small', ncol=2)

    # Global Title
    fig.suptitle(f"GLUE Sweep Results ({n_reps} reps)", fontsize=16)

    plt.savefig("./figS6.png")


if __name__=='__main__':
    main()