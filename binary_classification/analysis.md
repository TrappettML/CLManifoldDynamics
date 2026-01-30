# Summary
After training a learner and expert according to bc_code.md, we want to anlyze its performance. The first step is to to visualize the train/test loss and accuracy. We use the test accuracy or loss to compute the CL Metrics. The training code saves the weights and representations for every log_freq epoch out of all the training epochs for each task. These are both used in Plasticine analysis. Finally, glue uses the representations to calculate capacity, dim and radius for the class manifolds.

We want to call each of these in the single_run.py script. Theses analysis methods will read in the data saved from the main training script following the format found in bc_code.md. Each analysis will calculate its correpsonding metrics. The results for each analysis will be saved in a way so as to be used in a comparison plotter, where we can compare these metrics between algorithms. Right now We have only RL and SL but will add others in the future. 

# Overview
This document describes the three analysis pipelines that process continual learning artifacts to generate insights about plasticity, performance dynamics, and representational geometry.

# 1. CL Metrics Analysis (cl_analysis.py)
### Purpose:
Quantifies continual learning performance through various metrics based on accuracy or loss performance.
### Input Data
Data is saved in the single_run.py script after training. See bc_code.md and single_run.py for how data is saved. 

### Processing Steps

**1.1 Data Preparation**
* **Input:** Directory path to experiment: ```/(dataset)/(algorithm)/task_xxx/```
* **Parse Data:** Read data from ```(task_xxx)/metrics.pkl```; this contains acc/loss data for specific task. 
* **Combine Data:** Combine data into Matrix shape: ```(N_tasks_train, N_task_eval, epochs_per_task, Repeats) for each test acc and loss. 

**1.2 Performance Matrix Construction**
We construct the tensor $M$ where indices correspond to $(i, j, r)$.

* **Definition:** $M[i, j, r]$ represents the performance on **Task $j$** while training on **Task $i$** for **Repeat $r$**.
* **Integration:** Apply the selected method (e.g., AUC summation or indexing the Final value) over the `Steps_Per_Task` dimension.
* **Transpose:** Ensure dimensions align with the (Train, Eval) convention.
    * Final Shape: `(N_Train_Tasks, N_Evak_Tasks, N_Repeats)`

**1.3 Expert Vector Construction**
We compute the reference "Expert" performance vector $E$.

* **Definition:** $E[i, r]$ represents the ideal/expert performance on **Eval Task $i$** for **Repeat $r$**.
* **Integration:** Apply the *same* integration logic used for $M$ to the expert baselines (ensuring AUC comparisons are valid).
    * Final Shape: `(N_Eval_Tasks, N_Repeats)`

## Metrics and computation:
## I. Definitions & Notation

#### 1. Chance-Corrected Performance ($\hat{P}$)
Since this is a binary classification task, we normalize all raw accuracy measurements to account for the random baseline of 50%.

Let $A_{i,j}(t)$ denote the raw accuracy on **Evaluation Task $j$** while the model is training on **Current Task $i$** at epoch $t$. We define the **Chance-Corrected Performance** $\hat{P}$ as:

$$
\hat{P}_{i,j}(t) = 2 \cdot A_{i,j}(t) - 1
$$

* **$\hat{P} = 1.0$**: Perfect Accuracy (100%)
* **$\hat{P} = 0.0$**: Random Chance (50%)
* **$\hat{P} < 0.0$**: Adversarial / Worse than chance

### 2. Integration Methods (The $R$ Matrix)
We compute the Performance Matrix $R \in \mathbb{R}^{N \times N}$ (where $N$ is the number of tasks) using two distinct integration methods to capture different aspects of learning.

The matrix is oriented such that:
* **Columns ($i$)**: The task being **trained**.
* **Rows ($j$)**: The task being **evaluated**.

Consequently, an entry $R_{i,j}$ represents the performance on Task $j$ after (or during) the training of Task $i$.

**A. Final Performance Matrix ($R^{\text{final}}$)**
This metric captures the **capacity** or state of the model at the very end of a training block. It indexes the final value of the performance curve.

$$
R_{i,j}^{\text{final}} = \hat{P}_{i,j}(T_{\text{end}})
$$

**B. Dynamic (AUC) Matrix ($R^{\text{AUC}}$)**
This metric captures the **efficiency** and **consistency** of the model throughout the training block by computing the Area Under the Curve (AUC) of the performance trajectory.

$$
R_{i,j}^{\text{AUC}} = \int_{t=0}^{t=T^{j}_{\text{end}}} \hat{P}_{i,j}(t) dt
$$

where the integral is calculated using the trapezoid rule. 

---

## II. Matrix-Derived Metrics
These metrics are calculated using the entries of the $R$ matrix (either $R^{\text{final}}$ or $R^{\text{AUC}}$).

#### 1. Remembering / Stability Ratio (Lower Triangle)
*Relationship:* $j > i$ (e.g., Evaluating Task 0 after training Task 1)
Measures stability of previous knowledge. Normalized to be between $\pm$ 1. 

$$
\mathcal{R}_{i,j} = \frac{R_{j,i} - {R_{i,i}}}{R_{j,i} + {R_{i,i}}}
$$

* **Interpretation ($R^{\text{final}}$):** The classic "Stability" metric.
    * $\gt 1$: Backward Transfer.
    * $0.0$: Perfect Retention.
    * $\lt 1.0$: Forgetting due to interference.

* **Interpretation ($R^{\text{AUC}}$):** "Sustained Stability." If $\approx 0$, the model held the knowledge *throughout* the entire training block of the new task, not just at the end.

#### 2. Zero-Shot Transfer Ratio (Upper Triangle)
*Relationship:* $j < i$ (e.g., Evaluating Task 1 after training Task 0)
Measures how much of the future task is already solved before specific training begins.

$$
\mathcal{T}_{i,j} = \frac{R_{j,i} - {R_{i,i}}}{R_{j,i} + {R_{i,i}}}
$$
*(Note: Denominator $R_{i,i}$ represents the future performance on task $i$ once it is actually trained)*

* **Interpretation ($R^{\text{final}}$):** How much the network already knows for task $j$. 
* **Interpretation ($R^{\text{AUC}}$):** Unsure if specific interpretation. 

#### 3. Forward Transfer (Diagonal vs. Expert)
*Relationship:* $i = j$
Measures the learning quality of the current task relative to an independent "Expert" baseline ($E_i$).

$$
\text{FT}_{i} = \frac{E_{i} - R_{i,i}}{E_{i} + R_{i,i} + \epsilon}
$$
*(Note: $E_i$ must also be chance-corrected)*

* **Interpretation ($R^{\text{final}}$):** Did prior learning restrict the model's final capacity compared to a fresh model?
* **Interpretation ($R^{\text{AUC}}$):** Did prior learning accelerate convergence (higher AUC) compared to a fresh model?

This metric also captures plasticity, since we are comparing against a freshly initialized network. If the learner performs better then it could either mean better plasticity or transfer. We can control for this via continual backprop or episodic replay in future experiments. 

---

### III. Temporal Dynamics & Rate Analysis
These metrics analyze the shape of the learning curve $\hat{P}(t)$ directly, rather than just the integration.

#### 4. Effective Learning Time ($\tau$)
A participation ratio metric that characterizes the timescale of learning independent of total epochs.
Let $b(t) = \hat{P}(T_{end}) - \hat{P}(t)$ be the "unlearned" portion.

$$
\tau = \frac{\left[ \int_{0}^T b(t) \, dt \right]^2}{\int_{0}^T [b(t)]^2 \, dt}
$$

* **Interpretation:** Small $\tau$ indicates step-function-like (instant) learning. Large $\tau$ indicates slow, gradual acquisition.

#### 5. Forgetting Rate (Decay Slope)
Calculates the speed of forgetting for Task $i$ while Task $j$ is training.
$$
\text{Slope}_{j,i} = \frac{\text{Cov}(t, \hat{P}_{j,i}(t))}{\text{Var}(t)}
$$
* **Interpretation:** A negative slope indicates active forgetting. A zero slope indicates stability.

#### 6. Transient Instability (Max Dip)
Captures the worst-case performance on Task $i$ during the training of Task $j$, relative to its start.

$$
\text{Dip}_{j,i} = \max_{t} \left( | R_{i,i}^{\text{final}} - \hat{P}_{j,i}(t) | \right)
$$

---



## 1.4 Aggregation

avg_rem = nanmean(Rem)
std_rem = std(nanmean(Rem, axis=(0,1)))  # std across repeats
avg_ft = nanmean(FT)
std_ft = std(nanmean(FT, axis=1))  # std across repeats

Output
File: results/{dataset}/{algorithm}/cl_metrics.pkl
Structure:
python{
    'transfer': np.array,  # (N_Eval_Tasks, N_Repeats)
    'remembering': np.array,  # (N_Eval_Tasks, N_Train_Tasks, N_Repeats)
    'stats': {
        'rem_mean': float,
        'trans_mean': float
    }
}
Plots
None (metrics are summarized in terminal and saved to pickle)

# 2. Plasticity Analysis (plastic_analysis.py)

Purpose
Tracks representation and weight dynamics throughout continual learning to measure neural plasticity using metrics from the Chou et al. 2025 paper.
Input Data
Per Task Directory: results/{dataset}/{algorithm}/task_{t:03d}/

representations.npy: Hidden layer activations

Shape: (L, N_Repeats, N_Eval_Tasks, N_Subsamples, Hidden_Dim)
Where L = Total_Steps // log_frequency


weights.npy: Flattened model parameters

Shape: (L, N_Repeats, Param_Dim)

Global:

init_weights.npy: Initial random weights

Shape: (Param_Dim,)

Processing Steps
2.1 Per-Task Processing
For each task t in T tasks:
Load Data:

Load task_{t:03d}/representations.npy
Reshape to: (Steps, Repeats, Total_Samples, Dim) where Total_Samples = N_Eval_Tasks × N_Subsamples
Load task_{t:03d}/weights.npy

Reference Points:

init_w: Initial weights (same for all tasks)
task_ref_w: Final weights from previous task (or init_w for first task)

2.2 Representation Metrics (per step)
Computed via _compute_rep_metrics_batch (JIT compiled)
For each representation snapshot F: (Repeats, Samples, Dim)

Dormant Neurons Ratio

Compute average activation per neuron: avg_act[d] = mean(F[:,:,d])
Normalize: scores[d] = avg_act[d] / mean(avg_act)
Dormant: sum(scores <= tau) / Hidden_Dim where tau=0.1


Active Units Fraction

mean(mean(F > 0, axis=1)) across repeats and samples


Stable Rank

SVD: _, s, _ = svd(F_flat) where F_flat is (Repeats×Samples, Dim)
Cumulative energy: cum_energy = cumsum(s) / sum(s)
Stable rank: argmax(cum_energy > 0.99) + 1


Effective Rank

Entropy: H = -sum(p * log(p)) where p = s / sum(s)
Effective rank: exp(H)


Feature Norm

mean(norm(F, axis=-1)) per sample, then average


Feature Variance

mean(var(F, axis=0)) across neurons



Output per step: (6_metrics, N_Repeats)
2.3 Weight Metrics (per step)
Computed via _compute_weight_metrics_batch (JIT compiled)
For weights w: (N_Repeats, Param_Dim)

Weight Magnitude

sqrt(mean(w^2)) per repeat


Weight Difference (Task)

sqrt(mean((w - task_ref_w)^2)) per repeat


Weight Difference (Init)

sqrt(mean((w - init_w)^2)) per repeat



Output per step: (3_metrics, N_Repeats)
2.4 Temporal Aggregation

Stack metrics across all steps: (Total_Steps_Across_All_Tasks, N_Repeats)
Track task boundaries in epoch space

Output
File: results/{dataset}/{algorithm}/plastic_analysis_{dataset}.pkl
Structure:
```
python{
    'history': {
        'Dormant Neurons (Ratio)': np.array,  # (Total_Steps, N_Repeats)
        'Active Units (Fraction)': np.array,
        'Stable Rank': np.array,
        'Effective Rank': np.array,
        'Feature Norm': np.array,
        'Feature Variance': np.array,
        'Weight Magnitude': np.array,
        'Weight Difference (Task)': np.array,
        'Weight Difference (Init)': np.array
    },
    'task_boundaries': List[int]  # Cumulative epoch counts [E1, E1+E2, ...]
}
```


### Plots
**File:** `figures/{dataset}/{algorithm}/plasticine_metrics_{dataset}.png`

**Layout:** Vertical stack of 9 subplots (one per metric)
- **X-axis:** Epochs (continuous across tasks)
- **Y-axis:** Metric value
- **Lines:** Mean ± std across repeats
- **Markers:** Red vertical dashed lines at task boundaries

---

# 2. GLUE Analysis (`glue_analysis.py`)

### Purpose
Analyzes representational geometry evolution using manifold capacity theory (Chou et al. 2025). Quantifies how task-relevant manifolds untangle during learning.

### Input Data
**Per Task Directory:** `results/{dataset}/{algorithm}/task_{t:03d}/`
- `representations.npy`: 
  - Shape: `(L, N_Repeats, N_Eval_Tasks, N_Subsamples, Hidden_Dim)`
  
- `binary_labels.npy`: Class labels for analysis samples
  - Shape: `(N_Eval_Tasks, N_Subsamples, N_Repeats)`

### Theoretical Background

#### Manifold Capacity (Definition 2.1 from paper)
Measures how many class manifolds can be linearly separated in a representation space:
```
α_sim = P / n*
```
where n* is the minimum dimension needed for 50% separability probability

#### GLUE Metrics (Definition 2.3 from paper)
Effective geometric measures that explain capacity:

1. **Capacity** (α_M): Packing efficiency via mean-field approximation
2. **Dimension** (D_M): Degrees of freedom within manifolds (like Gaussian width)
3. **Radius** (R_M): Noise-to-signal ratio of manifolds
4. **Center Alignment** (ρ_c): Correlation between manifold centers
5. **Axis Alignment** (ρ_a): Correlation between manifold variation axes
6. **Center-Axis Alignment** (ψ): Cross-correlation between centers and axes

**Capacity Formula (Equation B.7):**
```
α_M ≈ (1 + R_M^(-2)) / D_M
```

### Processing Steps

#### 3.1 Initialization
- Create master PRNG key from config seed
- Initialize storage for 6 GLUE metrics + 2 extra (Simulated Capacity, Relative Error)

#### 3.2 Per-Task Processing
For each task t in T tasks:

**Load Data:**
- `reps_data: (Steps, Repeats, Samples, Dim)`
- `labels: (Samples, Repeats)` → reshape to `(Samples, Repeats)` if needed

**Key Management (for reproducibility):**
```
step_key = fold_in(master_key, task_idx)
step_key = fold_in(step_key, step)
unique_key = fold_in(step_key, repeat)
glue_key, sim_key = split(unique_key)
```

#### 3.3 Per-Step, Per-Repeat Computation

**For each step s, repeat r:**

**3.3.1 Data Subsampling**
- Select `config.analysis_subsamples` points per class (randomly shuffled)
- Ensures computational tractability for QP solver

**3.3.2 GLUE Metrics Computation**
Calls `run_manifold_geometry(glue_key, reps, labels, n_samples_t=config.n_t)`

**Algorithm:**
1. Group representations by class label
2. Sample n_t random directions `t ~ N(0, I)`
3. Sample n_t random dichotomies `y ~ {-1,+1}^P`
4. For each (t, y) pair:
   - Solve QP (Equation 2.2 via OSQP):
```
     max_{s_i ∈ M_i} ||proj_cone({y_i s_i}) t||²
```
   - Extract anchor points s_i(y,t) from dual variables
5. Decompose anchors: `s_i = s⁰_i + s¹_i(y,t)` (center + axis)
6. Compute geometry from anchor statistics:
   - **D_M:** Mean projection magnitude onto random directions
   - **R_M:** Ratio of total-to-axis projection norms
   - **ρ_c:** Pairwise cosine similarity of centers
   - **ρ_a:** Expected pairwise cosine similarity of axes
   - **ψ:** Expected center-axis cross-similarity
7. Capacity formula: `α_M ≈ (1 + R_M^(-2)) / D_M`

**Output:** 6 floats (one per GLUE metric)

**3.3.3 Simulated Capacity (every 100 epochs)**
Calls `compute_simulated_capacity(sim_key, reps, labels)`

**Algorithm 1 (Binary Search):**
1. Organize reps into P manifolds with M points each
2. Binary search for n* in [1, N_ambient]:
```
   For n_mid:
     - Estimate p_n via 100 trials of:
       a) Random project to n dimensions
       b) Random dichotomy
       c) Check linear separability (OSQP feasibility)
     - If p_n >= 0.5: search left (smaller n)
     - Else: search right (larger n)
```
3. Return `α_sim = P / n*`

**Relative Error:**
```
rel_error = |α_sim - α_M| / α_M
3.4 Aggregation

Stack results across steps: (Steps, N_Repeats) per metric
Concatenate across tasks: (Total_Steps, N_Repeats)
Track task boundaries and epochs for Simulated Capacity overlay

Output
File: results/{dataset}/{algorithm}/manifold_metrics.pkl
Structure:
python{
    'task_000': {
        'Capacity': np.array,              # (Steps, N_Repeats)
        'Radius': np.array,
        'Dimension': np.array,
        'Center_Alignment': np.array,
        'Axis_Alignment': np.array,
        'Center_Axis_Alignment': np.array,
        'Simulated_Capacity': np.array,    # (Steps, N_Repeats), sparse (every 100 epochs)
        'Capacity_Relative_Error': np.array
    },
    'task_001': { ... },
    ...
}
Plots
File: figures/{dataset}/{algorithm}/manifold_metrics_{dataset}.png
Layout: Vertical stack of 6 subplots (one per GLUE metric)
Per subplot:

X-axis: Epochs (continuous, steps * log_frequency)
Y-axis: Metric value
Main line: GLUE metric mean ± std (shaded region)
Overlay (Capacity subplot only):

Black scatter with error bars for Simulated Capacity (every 100 epochs)
Legend: "Simulated (Alg 1)"


Task boundaries: Red vertical dashed lines
Title (top plot): "Manifold Geometric Analysis - {dataset}"

Interpretation Guide:

↑ Capacity = Better untangling (higher packability)
↓ Radius → ↑ Capacity (smaller noise-to-signal)
↓ Dimension → ↑ Capacity (lower intrinsic dimensionality)
↓ Center/Axis Alignment → ↑ Capacity (less interference)
Simulated vs GLUE: Agreement validates mean-field approximation


Analysis Pipeline Integration in single_run.py
Execution Order

Training Loop → saves artifacts per task
CL Metrics → computes Remembering/Transfer
Plasticity Analysis → loads saved artifacts, computes/plots
GLUE Analysis → loads saved artifacts, computes/plots

Configuration Dependencies

config.log_frequency: Controls temporal resolution (L = epochs // log_freq)
config.analysis_subsamples: Points per class for GLUE (computational budget)
config.n_t: Number of random directions for GLUE (default ~50)
config.metric_type: 'acc' or 'loss' for CL metrics
config.m_integrator: 'auc', 'final', 'mean', etc. for CL metrics

