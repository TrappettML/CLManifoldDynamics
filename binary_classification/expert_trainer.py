import numpy as np
from learner import ContinualLearner

def train_single_expert(config, train_task, test_ds):
    """
    Trains n_repeats models on a single task from scratch.
    Returns: (loss_mean, loss_std, acc_mean, acc_std, all_losses, all_accs)
    """
    print(f"--- Training Expert on {train_task.name} ---")
    
    learner = ContinualLearner(config)
    
    # Preload data
    train_imgs, train_lbls = learner.preload_data(train_task.load_data())
    test_imgs, test_lbls = learner.preload_data(test_ds)
    
    final_accs = []
    final_losses = []
    
    for epoch in range(config.epochs_per_task):
        # Train
        learner.state, _, _ = learner._train_epoch_jit(
            learner.state, train_imgs, train_lbls
        )
        
        # Eval
        losses, accs = learner._eval_jit(
            learner.state, test_imgs, test_lbls
        )
        
        # Store for final epoch calculation
        final_accs.append(accs)
        final_losses.append(losses)

    # Calculate stats across the n_repeats for the final epoch
    loss_mean = np.mean(final_losses)
    loss_std = np.std(final_losses)
    acc_mean = np.mean(final_accs)
    acc_std = np.std(final_accs)

    print(f"    > Expert Final: Acc {acc_mean:.4f} | Loss {loss_mean:.4f}")
    return loss_mean, loss_std, acc_mean, acc_std, final_losses, final_accs