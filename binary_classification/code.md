# Project Summary

This project implements a scalable, high-performance Continual Learning (CL) framework for binary classification using JAX and Flax. The experiment consists of T sequential tasks, where each task discriminates between two specific classes sampled from a source dataset. To normalize against class similarity, R repeats are executed in parallel, with each repeat utilizing a unique, randomized pair of classes.

The architecture uses PyTorch for data loading and JAX for compiled, vectorized execution on GPUs. To support an arbitrary number of tasks, training and testing data are processed and loaded into JAX tensors immediately prior to each task execution. All data structures strictly adhere to specific format to facilitate vectorized scanning. Training artifacts—specifically model weights and representations—are accumulated in GPU memory and flushed to disk only during task transitions to optimize throughput.


### Experiment Architecture

Scope: $T$ Tasks; Binary Classification, $y \in \{0, 1\}$.

Parallelism: Execute R repeats (n_repeats) simultaneously using JAX vmap.

Task Definition:

At Task $t$, distinct pairs of classes ($C^A$,$C^B$) are sampled from the source dataset, in this case it will be imagenet-1k, cropped to 28x28 and gray scaled.

Per-Repeat Randomization: Class pairs are sampled independently for each repeat to average out class-similarity bias. Each repeat samples a random permutation of all available classes without replacement. For example, if we used MNIST (10 classes), each repeat can have at most T=5 tasks, where Repeat 1 might train on [(0,1), (2,3), (4,5), (6,7), (8,9)] while Repeat 2 trains on [(3,7), (0,5), (1,8), (2,6), (4,9)].

Binary Labeling: Within each task, $C^A$ always maps to label 0 and $C^B$ always maps to label 1.

Additionally, an "Expert" baseline is trained for every task. The Expert uses the same network architecture but is re-initialized from scratch at the start of each task and trained using the learning algorithm solely on the current task's data.

### Data Pipeline & Memory Management

Task Classes: Class pairs should be pre-computed for every task and every repeat before training begins.

Lazy Loading (Scalability): Training data for Task t should be generated/loaded to GPU VRAM immediately before Task t begins and cleared immediately after to support arbitrary T.

Test Data Pre-loading: When the number of tasks is small, test sets for all T tasks can be pre-loaded and kept in memory throughout the entire experiment to enable continuous evaluation on all tasks (for backward/forward transfer analysis).
However, when nTasks are large, we need to implement chunking so as to not OOM the gpu.

Tensor Layouts:

Storage/Canonical: (Total_Samples, N_Repeats, H, W). Putting samples first allows easier slicing, We will ned H,W for the images.

Training/Scan: (Num_Batches, Batch_Size, N_Repeats, H, W).

Weights (located in state): PyTree with leading dimensions (N_Repeats, ...). Individual weight arrays follow standard shapes (e.g., kernel: (Input_Dim, Output_Dim), bias: (Output_Dim,)). We will flatten all weights in a standard way.

Pre-Processing: We process all imagenet-1k data with prepare_imagenet.py. This gets the data from hugginface datasets, crops it to 28x28 and grays the images. It saves them using numpy. 

Expert Data Usage: The Expert utilizes the exact same Training/Scan tensors and Test set pre-loads already in VRAM for the main learner. No new data loading or duplication is required. The expert is evaluated only on the current task test data. 

### Training & Optimization (JAX/Flax)
Compilation: The training loop for a single task is fully compiled using jax.jit and jax.lax.scan.

Batching: The scan loop iterates over the Num_Batches dimension. Inside the scan, jax.vmap is used to parallelize operations over the N_Repeats axis.

Synchronized Logging: At log_frequency intervals, the system captures a synchronized snapshot of performance and state. To optimize for downstream vectorized analysis, all artifacts are stored as dense tensors with a leading time-series dimension (L=Total_Batches//log_freq). The log_frequency parameter can be adjusted to control memory usage if accumulated artifacts exceed available GPU memory.

Representations: Hidden states for S fixed subsamples from every task's test set.

Shape: (L, N_Repeats, N_Eval_Tasks, N_Subsamples, Hidden_Dim)

Note: S subsamples are drawn from each task's test set independently.

Binary Labels: Binary labels {0,1} aligned with representations.

Shape: (L, N_Repeats, N_Eval_Tasks, N_Subsamples)

Metrics: Evaluation accuracy and loss across all tasks.

Shape: (L, N_Repeats, N_Eval_Tasks)

Note: One for each loss and accuracy

Weights: Model parameters (PyTree) frozen at the log step.

Leading dimensions: (L, N_Repeats)

Internal structure: Preserved as nested PyTree (e.g., {'dense1': {...}, 'dense2': {...}})

Accumulation: These tensors are accumulated in device memory (VRAM) throughout the task execution and transferred to the host only after the task completes.

Expert Baseline Training:

Initialization: Fresh random initialization at the start of every task loop (transient state).

Optimization: Vanilla SGD via optax (no CL penalties or replay).

Parallelism: Uses jax.vmap over N_Repeats to train R independent experts simultaneously.

Expert Metrics: Tracks loss and accuracy for training and testing on the current task only.

Shape: (L, N_Repeats). Unlike the learner, it does not track performance across all T tasks, removing the N_Eval_Tasks dimension.

### I/O Strategy
Task Boundary I/O:

Train: Run task loop on GPU (accumulating logs).
Transfer: Move accumulated history (weights, representations, labels) from GPU to Host RAM.
Flush: Save history to disk with structured naming.
Clean: Delete task data from GPU to free space for Task t+1.

Saving Format & Directory Structure:
All training artifacts are organized hierarchically by dataset and algorithm:
```
results/
├── {dataset}/
│   ├── {algorithm}/
│   │   ├── task_001/
│   │   │   ├── representations.npy
│   │   │   ├── binary_labels.npy
│   │   │   ├── weights.npy
│   │   │   ├── metrics.pkl
│   │   │   ├── metadata.json
│   │   │   └── expert_metrics.pkl
```

Expert Metrics: expert_metrics.pkl stores a dictionary of arrays (train_loss, train_acc, test_loss, test_acc) with shape (L, N_Repeats), representing the expert's performance on the current task over time.

Naming Convention:

dataset: imagenet_28_gray
algorithm: Learning algorithm identifier (e.g., 'SL', 'RL', 'Bio', etc.) - the algorithm for computing error
    use_replay: Use an experience replay with cont.learner
    add_plasticity: Use shrink+perturb to induce plasticity
    use_ul: Use unsupervised learning in a layer or a step (unsure)
    num_tasks: how many tasks for experiment
    algorithm name will be a combination of all these or only learner_ntasks
task_{t:03d}: Zero-padded task index (e.g., task_000, task_001, ..., task_099)

File Formats:

Representations, Labels & Weights: NumPy ```.npy``` format for efficient array storage and loading
Metrics: Pickle ```.pkl``` format to store dictionary of metric arrays with metadata
Metadata: JSON ```.json``` format for human-readable configuration and reproducibility (includes task_class_pairs)

Example Save Path:
```results/imagenet_28_gray/SL/task_042/representations.npy```

## Learning Methods
### Architecture
I will be using a CNN with 3 CNN layers and 2 mlp layers. This seems to be a balanced net when compared in min_CNN_test. Learning happens relatively quickly but overfits quickly, ~35 epochs.

Maybe future work can compare architectures. 

### Learning Rules
The focus right now will be only SL. RL is implemented and will use a bio learning rule, maybe 3 factor.

### Experience Replay
Simple experience replay. 
#### Implementation:
I believe we can store a central replay buffer for all repeats, and then each replay can randomly sample for updating, this could provide memory benefits, but may increase computation time. 

### Plasticity
shrink+perturb

### Unsupervised Learning


### Data Anslysis

Data analysis will be later added as cl_analysis, plasticine_analysis, and glue_analysis. 
These analysis methods can be added to the single_run.py file at the end. This analysis code will load the data in from data saved from section 4. 