import ml_collections
import data_utils

def get_config():
    config = ml_collections.ConfigDict()
    
    config.dataset_name = 'fashion_mnist' 
    config.seed = 42
    config.data_dir = "./data"
    config.figures_dir = "./figures"
    config.reps_dir = "./saved_representations"
    config.num_tasks = 2
    
    config.input_dim = data_utils.get_dataset_dims(config.dataset_name)
    
    config.hidden_dim = 64
    config.learning_rate = 0.001
    config.batch_size = 128
    config.epochs_per_task = 10
    config.n_repeats = 10

    config.mandi_samples = 50 
    
    return config