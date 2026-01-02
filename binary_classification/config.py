import ml_collections
import data_utils

def get_config():
    config = ml_collections.ConfigDict()
    
    config.dataset_name = 'kmnist' 
    config.seed = 42
    config.data_dir = "./data"
    config.figures_dir = "./figures"
    config.reps_dir = "./saved_representations"
    config.num_tasks = 2
    down_sample = 15
    config.input_dim = int(down_sample**2) # data_utils.get_dataset_dims(config.dataset_name)
    config.downsample_dim = down_sample
    
    config.hidden_dim = 64
    config.learning_rate1 = 1e-2
    config.learning_rate2 = 1e-4
    config.batch_size = 128
    config.epochs_per_task = 1000
    config.n_repeats = 20

    config.mandi_samples = 50 
    config.eval_freq = 10  

    config.weight_decay = 0.0
    # --- Early Stopping Config ---
    config.early_stopping = False
    config.patience = 50          # Epochs to wait before stopping
    config.min_delta = 1e-4       # Minimum improvement required
    
    return config