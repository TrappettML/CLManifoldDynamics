import ml_collections
import os

def get_config(dataset_name, algorithm):
    config = ml_collections.ConfigDict()

    # Algorithm & Dataset
    config.algorithm = algorithm
    config.dataset_name = dataset_name
    config.seed = 42
    
    # Directory Structure (Aligned with Spec Section 4)
    # Spec: results/{dataset}/{algorithm}/task_001/
    config.data_dir = "./data"  # PyTorch dataset cache
    config.results_root = "results"
    config.results_dir = os.path.join("results", dataset_name, algorithm)
    
    # Plots go in a separate directory for convenience
    config.figures_dir = os.path.join("results", dataset_name, algorithm, "plots")

    # Task Configuration
    config.num_tasks = 2
    
    # Model & Data
    down_sample = 15
    config.input_dim = int(down_sample**2)
    config.downsample_dim = down_sample
    config.hidden_dim = 64
    
    # Optimization
    config.learning_rate1 = 1e-2  # Feature layers
    config.learning_rate2 = 1e-4  # Readout layer
    config.batch_size = 128
    config.weight_decay = 0.0
    
    # Training Schedule
    config.epochs_per_task = 1000
    config.log_frequency = 10
    config.n_repeats = 30
    
    if config.epochs_per_task % config.log_frequency != 0:
        raise ValueError("epochs_per_task must be divisible by log_frequency")
    
    # Early Stopping (Optional)
    config.early_stopping = False
    config.patience = 50
    config.min_delta = 1e-4
    
    # Analysis Parameters
    config.analysis_subsamples = 100
    config.n_t = 20
    config.metric_type = 'acc'
    config.m_integrator = 'final'

    return config