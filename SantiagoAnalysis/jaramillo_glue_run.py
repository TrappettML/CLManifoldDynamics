import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70" # set a hard cap on GPU mem

import numpy as np
import pickle
from analysis import load_data
import jax
import jax.numpy as jnp

print(jax.devices(), flush=True)

from glue_module.glue_analysis import run_glue_solver
from jaxopt import OSQP


def align_and_reshape_numpy(features, metadata):
    """
    NumPy alignment and reshaping (runs on CPU).
    """
    # 1. Apply sort order (Pure NumPy indexing)
    sorted_features = features[metadata['sort_indices']]
    
    # 2. Reshape
    return sorted_features.reshape(
        metadata['n_classes'], 
        metadata['samples_per_class'], 
        -1
    )


def get_sort_metadata(y):
    """
    Analyzes labels to create static metadata. 
    Returns numpy arrays to keep preprocessing on CPU.
    """
    y_cpu = np.array(y)
    
    unique_labels, counts = np.unique(y_cpu, return_counts=True)
    
    if len(np.unique(counts)) > 1:
        raise ValueError(f"Labels must be balanced. Found: {dict(zip(unique_labels, counts))}")
    
    # 1. Get sort indices
    sort_indices = np.argsort(y_cpu, kind='stable')
    
    # 2. Return dictionary (Keep sort_indices as NumPy array)
    return {
        'sort_indices': sort_indices, # Changed from jnp.array(sort_indices)
        'n_classes': int(len(unique_labels)),
        'samples_per_class': int(counts[0]),
        'unique_labels': unique_labels
    }



def run_glue_analysis(x_reshaped, key, qp):
    '''x_reshaped has shape: (P,M,N)'''
    new_key, next_key = jax.random.split(key)
    P = x_reshaped.shape[0]
    M = x_reshaped.shape[1]
    N = x_reshaped.shape[2]
    n_t = 200 # chou2025a uses n_t = 200
    geometries, _ = run_glue_solver(new_key, x_reshaped, P, M, N, n_t, qp)
    return geometries, next_key



def main():
    stim_types = ['AM', 'PT', 'NS']
    periods = ['base', 'onset', 'sustained', 'offset']
    regions = ['Primary auditory area',
            'Ventral auditory area',
            'Dorsal auditory area',
            'Posterior auditory area']

    n_units_min = 40  # number of units from each session and brain region to subsample for analysis
    qp = OSQP(tol=1e-4)
    
    # Compute some metric for all sessions, stimulus types, brain regions, and time periods.
    metric_avg = dict()
    metric_sem = dict()

    key = jax.random.PRNGKey(41)
    print("Starting Analysis", flush=True)
    n_subsample_repeats = 100
    sessionIDs = list(set(load_data('PT')['sessionIDArray']))

    for session in sessionIDs:
        print(f'Processing session {session}...', flush=True)
        for stim_type in stim_types:
            data = load_data(stim_type)
            
            # 1. Pre-calculate indices on CPU
            # We process masks here, but delay data loading until the period loop
            session_mask = data['sessionIDArray'] == session
            
            for region in regions:
                print(f"Processing region: {region}", flush=True)
                region_mask = data['brainRegionArray'] == region
                units_mask = np.logical_and(session_mask, region_mask)
                valid_indices = np.where(units_mask)[0]
                
                # Check sufficiency on CPU
                if len(valid_indices) >= n_units_min:
                    
                    for period in periods:
                        # 2. Load Data as NumPy (CPU)
                        x_full = np.array(data[period]) 
                        y = data['stimArray']
                        metadata = get_sort_metadata(y)

                        # 3. CPU Selection Loop
                        # Create the batch of subsamples and reshape them immediately on CPU
                        batch_reshaped_list = []
                        for _ in range(n_subsample_repeats):
                            # a. Subsample
                            chosen_idxs = np.random.choice(valid_indices, size=n_units_min, replace=False)
                            x_sub = x_full[:, chosen_idxs]
                            
                            # b. Align and Reshape (NumPy)
                            x_ready = align_and_reshape_numpy(x_sub, metadata)
                            batch_reshaped_list.append(x_ready)

                        # Stack and move to GPU
                        # Shape is now: (n_repeats, n_classes, samples_per_class, n_units)
                        batch_x_jax = jnp.array(np.stack(batch_reshaped_list))

                        # 4. JAX Execution
                        new_key, key = jax.random.split(key)
                        keys = jax.random.split(new_key, n_subsample_repeats)

                        # Define solver wrapper (No reshaping needed here anymore)
                        def run_solver_only(x_pre_shaped, k):
                            geoms, _ = run_glue_analysis(x_pre_shaped, k, qp)
                            return jnp.array(geoms)

                        # Run vmap over the already-shaped data
                        batch_results = jax.vmap(run_solver_only)(batch_x_jax, keys)

                        # Calculate metrics
                        mean_metrics = jnp.mean(batch_results, axis=0)
                        sem_metrics = jnp.std(batch_results, axis=0)

                        # Save results
                        metric_avg[(session, stim_type, region, period)] = np.array(mean_metrics)
                        metric_sem[(session, stim_type, region, period)] = np.array(sem_metrics)


    print(f"Saving means, sems", flush=True)
    # Save both dictionaries to a single file
    with open('./analysis_results.pkl', 'wb') as f:
        pickle.dump({'means': metric_avg, 'sems': metric_sem}, f)

    print(f"Finished Analysis and saved data", flush=True)    
    return


if __name__=='__main__':
    main()