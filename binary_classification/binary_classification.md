# Project Summary

This project implements a scalable, high-performance Continual Learning (CL) framework for binary classification using JAX and Flax. The experiment consists of T sequential tasks, where each task discriminates between two specific classes sampled from a source dataset. To normalize against class similarity, R repeats are executed in parallel, with each repeat utilizing a unique, randomized pair of classes.

The architecture uses PyTorch for data loading and JAX for compiled, vectorized execution on GPUs. To support an arbitrary number of tasks, training and testing data are processed and loaded into JAX tensors immediately prior to each task execution. All data structures strictly adhere to specific format to facilitate vectorized scanning. Training artifacts—specifically model weights and representations—are accumulated in GPU memory and flushed to disk only during task transitions to optimize throughput.

### 1. Experiment Architecture

Scope: $T$ Tasks; Binary Classification, $y \in {0,1}$.

Parallelism: Execute R repeats (n_repeats) simultaneously using JAX vmap.

Task Definition:

At Task $t$, distinct pairs of classes ($C^A$,$C^B$) are sampled from the source dataset.

Per-Repeat Randomization: Class pairs must be sampled independently for each repeat to average out class-similarity bias (e.g., Repeat 1 trains on 0 vs 1, Repeat 2 trains on 3 vs 7).

### 2. Data Pipeline & Memory Management

Task Classes should be pre-computed for every task and every repeat.

Lazy Loading (Scalability): To support an arbitrary number of tasks ($T \rightarrow \infty$), do not pre-load all tasks. Data for Task t should be generated/loaded to GPU VRAM immediately before Task t begins and cleared immediately after. 

Tensor Layouts:

Storage/Canonical: (Total_Samples, N_Repeats, Input_Dim). Putting samples first allows easier slicing.

Training/Scan: (Num_Batches, Batch_Size, N_Repeats, Input_Dim).

Weights (located in state): (N_Repeats, Input_Dim, Output_Dim).

Pre-Processing: All Torch-based loading, resizing, and flattening happens on the CPU; data is cast to JAX arrays before being pushed to the device.

### 3. Training & Optimization (JAX/Flax)
Compilation: The training loop for a single task is fully compiled using jax.jit and jax.lax.scan.

Batching: The scan loop iterates over the Num_Batches dimension. Inside the scan, jax.vmap is used to parallelize operations over the N_Repeats axis.

Synchronized Logging: At log_frequency intervals, the system captures a synchronized snapshot of performance and state. To optimize for downstream vectorized analysis, all artifacts are stored as dense tensors with a leading time-series dimension (L=Total_Batches//log_freq).

Representations: Hidden states for S fixed subsamples from every task's test set.

Shape: (L, N_Repeats, N_Eval_Tasks, N_Subsamples, Hidden_Dim)

Metrics: Evaluation accuracy and loss across all tasks.

Shape: (L, N_Repeats, N_Eval_Tasks)

Weights: Model parameters frozen at the log step.

Shape: (L, N_Repeats, Input_Dim, Output_Dim)

Accumulation: These tensors are accumulated in device memory (VRAM) throughout the task execution and transferred to the host only after the task completes.

### 4. I/O Strategy

Task Boundary I/O:

Train: Run task loop on GPU (accumulating logs).

Transfer: Move accumulated history (weights, representations) from GPU to Host RAM.

Flush: Save history to disk (pickle/numpys).

Clean: Delete task data from GPU to free space for Task t+1.