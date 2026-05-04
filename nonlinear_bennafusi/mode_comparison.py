import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ipdb import set_trace

# Import functions from the provided RSA framework
try:
    from RSA import (
        compare_representation_decompositions,
        compute_representation_decomposition,
        cumul_subspace_overlap,
    )
except ImportError:
    print("Error: RSA.py module not found. Ensure it is in the PYTHONPATH or the working directory.")
    sys.exit(1)


def load_task_representations(target_dir: Path) -> Dict[int, np.ndarray]:
    """
    Loads all task evaluation matrices (.npy) from the specified directory.
    Expected array shape per task: (N_TRIALS, NUM_CP, N_H, TEST_BATCH)
    """
    reps = {}
    files = list(target_dir.glob("eval_task_*.npy"))
    if not files:
        print(f"No files matching 'eval_task_*.npy' found in {target_dir}")
        sys.exit(1)
        
    for file_path in files:
        try:
            task_idx = int(file_path.stem.split("_")[-1])
            reps[task_idx] = np.load(file_path)
        except ValueError:
            continue
    return reps


def process_representation_pair(h_i: jnp.ndarray, h_j: jnp.ndarray, max_k: int) -> Tuple:
    """
    Computes decomposition and overlap metrics for a pair of representations.
    Returns the absolute similarity matrix and subspace overlap statistics.
    """
    _, _, U_i, n_sig_sigmas = compute_representation_decomposition(h_i.T) # transpose to get (nPoints,D_h) shape
    _, _, U_j, n_sig_sigmas = compute_representation_decomposition(h_j.T)

    # Absolute value resolves arbitrary sign assignments in SVD
    M_ij = jnp.abs(compare_representation_decompositions(U_i, U_j))
    # set_trace()
    overlaps, null_med, null_mean, p_vals, conf_bands = cumul_subspace_overlap(
        U_i, U_j, K=max_k, n_permutations=2000
    )
    return M_ij, overlaps, null_med, null_mean, p_vals, conf_bands


def compute_cross_task_metrics(reps: Dict[int, np.ndarray]) -> Tuple[Dict, Dict, List[int]]:
    """
    Calculates symmetric trial-averaged similarity matrices and overlap metrics 
    for the final checkpoint of all i > j task combinations.
    """
    task_indices = sorted(reps.keys())
    # Extract dimensions from the first available matrix
    n_trials = reps[task_indices[0]].shape[0]
    n_h = reps[task_indices[0]].shape[2]
    M_dict = {}
    overlap_dict = {}
    
    # JIT compile the core processing block to optimize the trial loop
    process_jitted = jax.jit(process_representation_pair, static_argnames=['max_k'])
    
    for i in task_indices:
        for j in task_indices:
            if i <= j:
                continue
            
            t_M, t_overlaps, t_n_med, t_n_mean = [], [], [], []
            t_p_vals, t_c_lower, t_c_upper = [], [], []
            
            for trial in range(n_trials):
                # Extract representation at the final checkpoint index (-1)
                h_i = jnp.array(reps[i][trial, -1, :, :])
                h_j = jnp.array(reps[j][trial, -1, :, :])
                
                M_ij, overlaps, null_med, null_mean, p_vals, conf_bands = process_jitted(
                    h_i, h_j, max_k=20
                )
                # set_trace()
                t_M.append(M_ij)
                t_overlaps.append(overlaps)
                t_n_med.append(null_med)
                t_n_mean.append(null_mean)
                t_p_vals.append(p_vals)
                t_c_lower.append(conf_bands[0])
                t_c_upper.append(conf_bands[1])
            
            M_dict[(i, j)] = np.mean(t_M, axis=0)
            overlap_dict[(i, j)] = {
                'overlaps': np.mean(t_overlaps, axis=0),
                'null_med': np.mean(t_n_med, axis=0),
                'null_mean': np.mean(t_n_mean, axis=0),
                'p_vals': np.mean(t_p_vals, axis=0),
                'conf_lower': np.mean(t_c_lower, axis=0),
                'conf_upper': np.mean(t_c_upper, axis=0),
            }
            
    return M_dict, overlap_dict, task_indices


def plot_similarity_matrix_grid(M_dict: Dict, task_indices: List[int], out_dir: Path):
    """Generates the lower triangular subplot matrix of the task comparisons."""
    n_tasks = len(task_indices)
    fig, axes = plt.subplots(n_tasks, n_tasks, figsize=(3 * n_tasks, 3 * n_tasks))
    
    # Handle scalar axes if n_tasks=1 (fallback)
    if n_tasks == 1:
        axes = np.array([[axes]])

    for r_idx, i in enumerate(task_indices):
        for c_idx, j in enumerate(task_indices):
            ax = axes[r_idx, c_idx]
            if i > j:
                im = ax.imshow(M_dict[(i, j)], cmap='viridis', vmin=0, vmax=1)
                ax.set_title(f"Task {i} vs Task {j}")
                if r_idx == n_tasks - 1:
                    ax.set_xlabel("Task j Modes")
                if c_idx == 0:
                    ax.set_ylabel("Task i Modes")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(out_dir / "similarity_matrix_grid.pdf")
    plt.close()


def plot_overlap_trace_grid(overlap_dict: Dict, task_indices: List[int], n_h: int, out_dir: Path):
    """Generates the lower triangular subplot grid for subspace overlaps across tasks."""
    n_tasks = len(task_indices)
    fig, axes = plt.subplots(n_tasks, n_tasks, figsize=(3 * n_tasks, 3 * n_tasks))
    x_axis = np.arange(1, n_h + 1)
    
    if n_tasks == 1:
        axes = np.array([[axes]])

    for r_idx, i in enumerate(task_indices):
        for c_idx, j in enumerate(task_indices):
            ax = axes[r_idx, c_idx]
            if i > j:
                data = overlap_dict[(i, j)]
                ax.plot(x_axis, data['overlaps'], label='True Overlap', color='black', lw=1.5)
                ax.plot(x_axis, data['null_med'], label='Null Median', color='red', ls='--')
                ax.fill_between(x_axis, data['conf_lower'], data['conf_upper'], color='red', alpha=0.15, label='95% CI')
                
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Task {i} vs Task {j}")
                if r_idx == n_tasks - 1:
                    ax.set_xlabel("Mode index (k)")
                if c_idx == 0:
                    ax.set_ylabel("Cumulative Overlap")
                    
                if r_idx == 1 and c_idx == 0:
                    ax.legend(loc='best', fontsize='small')
            else:
                ax.axis('off')
    
    plt.tight_layout()
    fig.savefig(out_dir / "overlap_trace_grid.pdf")
    plt.close()


def plot_global_averages(M_dict: Dict, overlap_dict: Dict, n_h: int, out_dir: Path):
    """Calculates and visualizes the grand average of M and overlaps across all task pairs."""
    if not M_dict:
        return
        
    # Aggregate data
    M_values = list(M_dict.values())
    avg_M = np.mean(M_values, axis=0)
    
    avg_overlaps = np.mean([v['overlaps'] for v in overlap_dict.values()], axis=0)
    avg_null_med = np.mean([v['null_med'] for v in overlap_dict.values()], axis=0)
    avg_conf_lower = np.mean([v['conf_lower'] for v in overlap_dict.values()], axis=0)
    avg_conf_upper = np.mean([v['conf_upper'] for v in overlap_dict.values()], axis=0)

    # Plot Average Matrix M
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(avg_M, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f"Mean Similarity |M| (Averaged over {len(M_values)} task pairs)")
    ax.set_xlabel("Task j Modes")
    ax.set_ylabel("Task i Modes")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "average_similarity_matrix.pdf")
    plt.close()

    # Plot Average Overlaps
    fig, ax = plt.subplots(figsize=(7, 5))
    x_axis = np.arange(1, n_h + 1)
    
    ax.plot(x_axis, avg_overlaps, label='True Overlap (Mean)', color='black', lw=2)
    ax.plot(x_axis, avg_null_med, label='Null Median (Mean)', color='red', ls='--')
    ax.fill_between(x_axis, avg_conf_lower, avg_conf_upper, color='red', alpha=0.2, label='Mean 95% CI')
    
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Mean Subspace Overlap")
    ax.set_xlabel("Mode index (k)")
    ax.set_ylabel("Cumulative Subspace Overlap")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "average_subspace_overlap.pdf")
    plt.close()


def run_pipeline(target_path: str):
    base_dir = Path(target_path)
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: Invalid directory path -> {target_path}")
        sys.exit(1)

    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading representations from: {base_dir}")
    reps = load_task_representations(base_dir)
    n_h = reps[list(reps.keys())[0]].shape[2]
    
    print(f"Computing pairwise metrics over {len(reps)} tasks...")
    M_dict, overlap_dict, task_indices = compute_cross_task_metrics(reps)
    
    if not M_dict:
        print("Not enough tasks strictly > 1 evaluated to perform lower-triangular cross-comparisons.")
        sys.exit(0)

    print(f"Generating visualizations. Saving to {plots_dir}...")
    plot_similarity_matrix_grid(M_dict, task_indices, plots_dir)
    plot_overlap_trace_grid(overlap_dict, task_indices, 20, plots_dir)
    plot_global_averages(M_dict, overlap_dict, 20, plots_dir)
    print("Execution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize representation alignments across tasks.")
    parser.add_argument(
        "target_path",
        type=str,
        help="Path to the task directory containing eval_task_*.npy files (e.g. /home/users/.../results/no-BF/task_1/)"
    )
    args = parser.parse_args()
    
    run_pipeline(args.target_path)