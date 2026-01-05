import ml_collections

dataset = 'fashion_mnist'

def get_config():
    config = ml_collections.ConfigDict()
    
    # --- New: Algorithm Selector ---
    config.algorithm = 'SL'  # Options: 'SL', 'RL', 'UL', etc.
    
    config.dataset_name = dataset
    config.seed = 42
    config.data_dir = "./data"
    config.figures_dir = "./figures"
    config.reps_dir = "./saved_representations"
    
    config.num_tasks = 2
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

    # mandi variables
    config.mandi_samples = 50 
    config.n_t = 50

    config.weight_decay = 0.0
    config.early_stopping = False
    config.patience = 50          
    config.min_delta = 1e-4    

    config.metric_type = 'acc'
    config.m_integrator = 'final'
    
    return config