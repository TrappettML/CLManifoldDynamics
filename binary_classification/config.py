import ml_collections
import os
import math
import jax
import socket

def get_config(algorithm, 
                use_replay=False, 
                add_plasticity=False, 
                use_ul=False, 
                num_tasks=2, 
                dataset_name="imagenet_28_gray", 
                lr1=1e-2, 
                lr2=1e-4,
                num_epochs=1000,
                outdim=1,
                ) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Store boolean flags
    config.use_replay = use_replay
    config.add_plasticity = add_plasticity
    config.use_ul = use_ul

    # Construct the base algorithm name based on flags
    algo_name = algorithm + f"_{num_tasks}_tasks"
    if config.use_replay:
        algo_name += "_er"
    if config.add_plasticity:
        algo_name += "_pl"
    if config.use_ul:
        algo_name += "_ul"

    algo_name += f"_epochs_{num_epochs}_lr1_{lr1}_lr2_{lr2}_dout_{outdim}"

    config.algorithm = algo_name
    config.dataset_name = dataset_name
    config.seed = 42

    cs_name = socket.gethostname()
    if "talapas" in cs_name:
        root = "/home/mtrappet/tau/manifold/binary_classification"
        config.data_dir = f"{root}/data/{dataset_name}"
        config.results_root = f"{root}/results"
        print(f"Saving results to ")
    else:
        config.data_dir = f"./data/{dataset_name}"
        config.results_root = "results"
    
    config.results_dir = os.path.join(config.results_root, dataset_name, config.algorithm)
    config.figures_dir = os.path.join(config.results_root, dataset_name, config.algorithm, "plots")

    config.num_tasks = num_tasks
    config.input_side = 28
    config.hidden_dim = 64
    config.output_dim = outdim
    
    config.learning_rate1 = lr1
    config.learning_rate2 = lr2
    config.batch_size = 128
    config.weight_decay = 0.0
    
    config.epochs_per_task = num_epochs
    config.log_frequency = 10
    
    # --- DYNAMIC REPEATS CALCULATION ---
    num_devices = jax.local_device_count()
    base_repeats = 32
    # Ensure repeats are divisible by number of GPUs (e.g. 3 GPUs -> 33 repeats)
    config.n_repeats = math.ceil(base_repeats / num_devices) * num_devices
    # -----------------------------------
    
    if config.epochs_per_task % config.log_frequency != 0:
        raise ValueError("epochs_per_task must be divisible by log_frequency")
    
    config.early_stopping = False
    config.patience = 50
    config.min_delta = 1e-4
    
    config.analysis_subsamples = 200
    config.n_t = 50
    config.metric_type = 'acc'
    config.m_integrator = 'final'

    return config