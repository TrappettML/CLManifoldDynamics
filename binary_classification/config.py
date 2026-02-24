import ml_collections
import os

def get_config(algorithm, use_replay=False, add_plasticity=False, use_ul=False, dataset_name="imagenet_28_gray") -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    # Store boolean flags
    config.use_replay = use_replay
    config.add_plasticity = add_plasticity
    config.use_ul = use_ul

    # Construct the base algorithm name based on flags
    algo_name = algorithm
    if config.use_replay:
        algo_name += "_er"
    if config.add_plasticity:
        algo_name += "_pl"
    if config.use_ul:
        algo_name += "_ul"

    # Algorithm & Dataset
    config.algorithm = algo_name
    config.dataset_name = dataset_name
    config.seed = 42
    
    # Directory Structure (Aligned with Spec Section 4)
    # Spec: results/{dataset}/{algorithm}/task_001/
    config.data_dir = f"./data/{dataset_name}"  # PyTorch dataset cache
    config.results_root = "results"
    
    # These will now dynamically use the new appended algorithm name (e.g., SL_er_pl)
    config.results_dir = os.path.join("results", dataset_name, config.algorithm)
    config.figures_dir = os.path.join("results", dataset_name, config.algorithm, "plots")

    # Task Configuration
    config.num_tasks = 2
    
    # Model & Data
    config.input_dim = 28*28 # using imagenet downsampled to 28x28
    config.hidden_dim = 64
    
    # Optimization
    config.learning_rate1 = 1e-2  # Feature layers
    config.learning_rate2 = 1e-4  # Readout layer
    config.batch_size = 128
    config.weight_decay = 0.0
    
    # Training Schedule
    config.epochs_per_task = 100
    config.log_frequency = 10
    config.n_repeats = 30
    
    if config.epochs_per_task % config.log_frequency != 0:
        raise ValueError("epochs_per_task must be divisible by log_frequency")
    
    # Early Stopping (Optional)
    config.early_stopping = False
    config.patience = 50
    config.min_delta = 1e-4
    
    # Analysis Parameters
    config.analysis_subsamples = 200
    config.n_t = 100
    config.metric_type = 'acc'
    config.m_integrator = 'final'

    return config