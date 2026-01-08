import ml_collections
import os

def get_config(dataset_name='kmnist', algorithm='SL'):
     config = ml_collections.ConfigDict()

     # --- Algorithm Selector ---
     config.algorithm = algorithm  # Options: 'SL', 'RL', 'UL', etc.

     config.dataset_name = dataset_name
     config.seed = 42
     
     # --- Directory Structure Refactor ---
     # Root: ./single_runs/(ALGORITHM)/
     algo_dir = os.path.join("./single_runs", config.algorithm.upper())
     
     config.data_dir = "./data"
     # Plot output: ./single_runs/SL/plots/
     config.figures_dir = os.path.join(algo_dir, "plots")
     # Data output (reps, analysis pkls): ./single_runs/SL/data/
     config.reps_dir = os.path.join(algo_dir, "data")

     config.num_tasks = 2
     
     # Note: The data loader forces grayscale, so input_dim is H*W
     down_sample = 15
     config.input_dim = int(down_sample**2) 
     config.downsample_dim = down_sample

     config.hidden_dim = 64
     config.learning_rate1 = 1e-2
     config.learning_rate2 = 1e-4
     config.batch_size = 128

     # --- Optimization & Logging ---
     config.epochs_per_task = 1000
     config.log_frequency = 10 
     config.n_repeats = 20
     config.eval_freq = 10 

     if config.epochs_per_task % config.log_frequency != 0:
          raise ValueError("epochs_per_task must be divisible by log_frequency")
     
     # early stopping and learning hyprms
     config.weight_decay = 0.0
     config.early_stopping = False
     config.patience = 50          
     config.min_delta = 1e-4    
     
     # mandi variables
     config.mandi_samples = 50 
     config.n_t = 20
     
     config.metric_type = 'acc'
     config.m_integrator = 'final'

     return config