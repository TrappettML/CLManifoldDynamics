"""
Continual learning with a 2‑layer *non‑linear* network trained by SGD.
 Replaces the Benna‑Fusi ODE experiments while keeping the
task‑boundary reset protocol – now realised as different weight‑transfer
strategies.

Conditions (analogous to the BF reset specs)
--------------------------------------------
- "no_reset"     → fine‑tune    : keep both W₁ and W₂ from previous task
- "feature_xfer" → keep W₁, randomly reinitialise W₂
- "slow_xfer"    → EMA of W₁ (slow feature extractor), random W₂
- "fresh"        → fully random initialisation (no transfer)

For each task the target is Σ = U₀ diag(s) Vᵏᵀ, with fixed U₀ and
task‑specific Vᵏ obtained by rotating a shared V₀ using the hybrid
variance model (as in dln_transfer.py).  Training minimises
  Eₓ[½‖ŷ(x) − Σ x‖²]   where   ŷ(x) = W₂ ReLU(W₁ x).

Metrics reported:
  - forward transfer efficiency (relative to zero‑weight baseline)
  - cross‑task alignment of the right singular vectors of Cov(ŷ, x)
  - learning curves and final loss
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from scipy.linalg import expm

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ======================================================================
#  Parameters (kept from the original BF experiment, adapted where needed)
# ======================================================================
N_Y = 10            # output dim  (= number of modes)
N_X = 30            # input dim   (≥ M_EXT)
N_H = 10            # hidden dim  (= N_Y)

M_EXT = 30          # extended rotation dimension
SIGMA_0 = 1.0
P_EXP   = 1.0
SIGMA_BAND = 1.0
BW      = 1.0
ALPHA_0 = 0.0

N_TASKS    = 20     # tasks per sequence
N_TRIALS   = 10     # independent sequences (random seeds)
BATCH_SIZE = 128
STEPS_PER_TASK = 2000   # SGD steps per task
LEARNING_RATE  = 1e-3

# Spectra (from dln_transfer.py)
def make_spectrum(n: int) -> np.ndarray:
    return np.exp(-2.0 * np.arange(n) / n)

S = make_spectrum(N_Y)    # shape (N_Y,)

# Seed bases
SEED_V0       = 0
SEED_ENSEMBLE = 42

# ======================================================================
#  Helper: hybrid variance & task rotation (re‑used from dln_transfer.py)
# ======================================================================
def _hybrid_variance_matrix(M: int) -> np.ndarray:
    alpha = np.arange(M)
    g = ((alpha + 1 + ALPHA_0) / (M + ALPHA_0)) ** P_EXP
    s2 = SIGMA_0**2 * g[:, None] * g[None, :]
    diff = alpha[:, None] - alpha[None, :]
    s2 += SIGMA_BAND**2 * np.exp(-diff**2 / (2 * BW**2))
    np.fill_diagonal(s2, 0.0)
    return s2

def _sample_rotation(s2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = s2.shape[0]
    std = np.sqrt(s2)
    A = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    A[iu] = std[iu] * rng.standard_normal(len(iu[0]))
    A = A - A.T
    return expm(A)

def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    H = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(H)
    return Q @ np.diag(np.sign(np.diag(R)))

def build_fixed_U_tasks(
    n_tasks: int,
    rng: np.random.Generator,
    V0_ext: np.ndarray,
    s: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    """Generate task targets Σ_k = U0 diag(s) V_k^T.

    V_k = V0_ext[:, :K] @ Q_k[:, :K]  with Q_k a random rotation of M_EXT.
    U0 is fixed (random orthogonal) to prevent selectivity artefacts.
    """
    K = len(s)
    s2 = _hybrid_variance_matrix(M_EXT)
    U0 = _random_orthogonal(N_Y, rng)
    tasks = []
    for _ in range(n_tasks):
        Q = _sample_rotation(s2, rng)
        Vk = (V0_ext @ Q)[:, :K]
        Sigma_k = U0 @ np.diag(s) @ Vk.T
        tasks.append({"Sigma": Sigma_k, "s": s, "U": U0, "V": Vk})
    return tasks, U0

# ======================================================================
#  Network and training (JAX)
# ======================================================================
class NetworkState(NamedTuple):
    W1: jnp.ndarray  # (N_H, N_X)
    W2: jnp.ndarray  # (N_Y, N_H)

def init_network(rng: jax.random.PRNGKey) -> NetworkState:
    k1, k2 = jax.random.split(rng)
    W1 = jax.random.normal(k1, (N_H, N_X)) * 1e-2
    W2 = jax.random.normal(k2, (N_Y, N_H)) * 1e-2
    return NetworkState(W1, W2)

def forward(state: NetworkState, x: jnp.ndarray) -> jnp.ndarray:
    """ŷ(x) = W₂ ReLU(W₁ x)"""
    return (state.W2 @ jax.nn.relu(state.W1 @ x.T)).T # y_true is BatchSize,D_out

@jax.jit
def loss_fn(state: NetworkState, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """MSE loss for a batch."""
    y_hat = forward(state, x)
    return 0.5 * jnp.mean(jnp.sum((y - y_hat) ** 2, axis=-1)) # average over batch dim

def train_on_task(
    init_state: NetworkState,
    opt_state,
    optimiser,
    Sigma: jnp.ndarray,
    rng: jax.random.PRNGKey,
    steps: int = STEPS_PER_TASK,
    batch_size: int = BATCH_SIZE,
) -> tuple[NetworkState, jnp.Array, dict]:
    """Run SGD on a single task, returning final state and diagnostics."""
    @jax.jit
    def step_fn(state, opt_state, batch):
        def single_batch_loss(params, x, y):
            return loss_fn(params, x, y)
        loss, grads = jax.value_and_grad(single_batch_loss)(state, *batch)
        updates, opt_state = optimiser.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss

    # Pre‑generate all batches (vmap over steps)
    key, subkey = jax.random.split(rng)
    keys = jax.random.split(subkey, steps)
    # Each batch: x ~ N(0,I), y = Sigma x
    def gen_batch(k):
        x = jax.random.normal(k, (batch_size, N_X))
        y = x @ Sigma.T   # (batch, N_Y)
        return x, y
    batches = jax.vmap(gen_batch)(keys)

    def scan_fn(carry, batch):
        state, opt_state = carry
        state, opt_state, loss = step_fn(state, opt_state, batch)
        return (state, opt_state), loss

    (final_state, final_opt_state), losses = jax.lax.scan(
        scan_fn, (init_state, opt_state), batches
    )

    # Estimate initial loss on a fresh batch
    init_batch = gen_batch(key)
    init_loss = loss_fn(init_state, *init_batch)

    return final_state, final_opt_state, {
        "init_loss": init_loss,
        "final_loss": losses[-1],
        "loss_curve": losses,
    }

# ======================================================================
#  Condition definitions (mimicking the BF “reset” specifications)
# ======================================================================

class Condition(NamedTuple):
    label: str
    # Function that, given previous network state (or None for first task),
    # returns (new state, opt_state, any persistent aux state)
    # We also pass an optional “slow” EMA of W1.
    init_fn: callable
    color: str
    linestyle: str

def _init_opt_state(optimiser, state):
    return optimiser.init(state)

# common optimiser
optimiser = optax.adam(LEARNING_RATE)

def fresh_init(rng, _prev_state, **_kwargs):
    """Random initialisation."""
    state = init_network(rng)
    opt_state = _init_opt_state(optimiser, state)
    return state, opt_state, None

def fine_tune_init(rng, prev_state, **_kwargs):
    """Continue from previous weights."""
    if prev_state is None:
        return fresh_init(rng, None)
        
    state = prev_state
    opt_state = _init_opt_state(optimiser, state)
    return state, opt_state, None

def feature_xfer_init(rng, prev_state, **_kwargs):
    """Keep W1, randomly reinitialise W2."""
    if prev_state is None:
        return fresh_init(rng, None)
    new_W2 = jax.random.normal(rng, (N_Y, N_H)) * 1e-2
    state = NetworkState(W1=prev_state.W1, W2=new_W2)
    opt_state = _init_opt_state(optimiser, state)
    return state, opt_state, None

def slow_xfer_init(rng, prev_state, ema_decay=0.995, ema=None):
    """Keep ‘slow’ W1 (EMA), random W2.  ema is a tuple: (ema_W1,)."""
    key1, key2 = jax.random.split(rng)
    if ema is None:
        state, opt_state, _ = fresh_init(key1, prev_state)
        ema = state.W1          # the starting slow weight
        opt_state = _init_opt_state(optimiser, state)
        return state, opt_state, ema
    ema_W1 = ema  # (N_H, N_X) array
    # Update EMA with current W1
    new_ema = ema_decay * ema_W1 + (1 - ema_decay) * prev_state.W1
    state = NetworkState(W1=new_ema, W2=jax.random.normal(key2, (N_Y, N_H)) * 1e-2)
    opt_state = _init_opt_state(optimiser, state)
    return state, opt_state, new_ema

# Conditions derived from the original BF table
CONDITIONS = [
    Condition("fine‑tune",   fine_tune_init,   "C7", ":"),
    Condition("feature xfer", feature_xfer_init, "C2", "--"),
    Condition("slow xfer",   slow_xfer_init,   "C0", "-"),
    Condition("fresh",       fresh_init,        "k",  ":"),
]

# ======================================================================
#  Trial‑level logic (one sequence of tasks, one random seed)
# ======================================================================
def run_trial(
    rng: jax.random.PRNGKey,
    tasks: list[dict],
    condition: Condition,
) -> dict:
    """Execute a complete multi‑task trial for one condition."""
    n_tasks = len(tasks)

    # Extract Sigma matrices and V matrices for alignment plotting
    Sigmas = jnp.stack([jnp.asarray(t["Sigma"]) for t in tasks])
    V_true  = jnp.stack([jnp.asarray(t["V"]) for t in tasks])  # (n_tasks, N_X, N_Y)

    # We will collect initial loss, final loss, and extracted V_emp
    init_losses = jnp.zeros(n_tasks)
    final_losses = jnp.zeros(n_tasks)
    V_emp = jnp.zeros((n_tasks, N_Y, N_X))

    state = None
    aux = None   # persistent aux (e.g., EMA for slow_xfer)
    opt_state = None

    for k in range(n_tasks):
        rng, init_rng, train_rng, extract_rng = jax.random.split(rng, 4)
        if k == 0:
            # First task – no previous state
            init_state, opt_state, aux = condition.init_fn(
                init_rng, None, ema=aux
            )
        else:
            init_state, opt_state, aux = condition.init_fn(
                init_rng, state, ema=aux
            )

        # Train
        final_state, opt_state, diag = train_on_task(
            init_state, opt_state, optimiser, Sigmas[k], train_rng
        )
        init_losses = init_losses.at[k].set(diag["init_loss"])
        final_losses = final_losses.at[k].set(diag["final_loss"])

        # Extract empirical V from final network
        # Use a large batch to approximate Cov(ŷ, x)
        x_big = jax.random.normal(extract_rng, (2048, N_X))
        y_hat_big = forward(final_state, x_big)
        cov = (y_hat_big.T @ x_big) / 2048   # (N_Y, N_X)
        _, _, Vh = jnp.linalg.svd(cov, full_matrices=False)
        V_emp = V_emp.at[k].set(Vh)  # right singular vectors

        state = final_state

    # Compute forward transfer efficiency (as in original code)
    # null loss: loss if output is zero: 0.5 * E[‖Σ x‖²] = 0.5 * ‖Σ‖_F²
    null_losses = 0.5 * jnp.sum(Sigmas**2, axis=(1,2))
    ft = 1.0 - init_losses[1:] / null_losses[1:]   # skip task 0

    return {
        "init_loss": init_losses,
        "final_loss": final_losses,
        "ft": ft,
        "V_emp": V_emp,
        "m": 1,   # dummy, for compatibility with plotting
    }

# ======================================================================
#  Alignment plotting (as requested)
# ======================================================================
def compute_alignment_matrix(V_emp):
    """Task‑averaged absolute mode overlaps."""
    num_tasks, K, _ = V_emp.shape
    # dots = jnp.abs(jnp.einsum('kai, lbj -> klab', V_emp, V_emp))
    dots = jnp.abs(jnp.einsum('kai, lbi -> klab', V_emp, V_emp))
    mask = 1.0 - jnp.eye(num_tasks)
    alignment = jnp.einsum('klab, kl -> ab', dots, mask) / (num_tasks * (num_tasks - 1))
    return alignment

def plot_alignment(alignment_matrix, out_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(alignment_matrix, cmap='viridis', origin='upper', vmin=0, vmax=1)
    plt.colorbar(label='Average Absolute Dot Product')
    plt.xlabel(r'Mode $\beta$ (Task $k^{\prime}$)')
    plt.ylabel(r'Mode $\alpha$ (Task $k$)')
    plt.title(r'Cross‑Task Mode Alignment $\langle v_\alpha^{(k)}, v_\beta^{(k^\prime)} \rangle$')
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def plot_individual_alignments(V_emp, out_path):
    n_tasks = V_emp.shape[0]
    fig, axes = plt.subplots(n_tasks, n_tasks, figsize=(2 * n_tasks, 2 * n_tasks))
    if n_tasks == 1:
        axes = np.array([[axes]])
    for i in range(n_tasks):
        for j in range(n_tasks):
            if j > i:
                axes[i][j].axis('off')
                continue
            M_ab = jnp.abs(V_emp[i] @ V_emp[j].T)
            im = axes[i][j].imshow(M_ab, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
            axes[i][j].set_title(f"M  T:{i},{j}")
            if i == n_tasks - 1:
                axes[i][j].set_xlabel(r'Mode $\beta$')
            if j == 0:
                axes[i][j].set_ylabel(r'Mode $\alpha$')
    plt.tight_layout()
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Absolute Magnitude')
    plt.savefig(out_path, dpi=100)
    plt.close()


# ======================================================================
#  Main experiment
# ======================================================================
def run_2l_nl():
    print("=" * 65)
    print("Non‑linear CL with weight‑transfer strategies (JAX SGD)")
    print("=" * 65)

    # ---- Generate the shared reference basis & spectrum ----
    rng_v0 = np.random.default_rng(SEED_V0)
    V0_ext, _ = np.linalg.qr(rng_v0.standard_normal((N_X, M_EXT)))
    print(f"Spectrum: s = {S.round(3)}")

    # ---- For each trial, we will run all conditions ----
    # We will vmap over trials and over conditions, yielding
    # results[trial, cond] of type dict.
    # We'll need a function that takes a trial key and returns a
    # dictionary of per‑condition results.
    def run_all_conds_one_trial(trial_rng: jax.random.PRNGKey,
                                trial_idx: int) -> dict:
        """Run one trial (same task sequence for all conditions)."""
        # Generate fresh task sequence for this trial
        rng_ens = np.random.default_rng(SEED_ENSEMBLE + trial_idx)
        tasks_np, U0 = build_fixed_U_tasks(N_TASKS, rng_ens, V0_ext, S)
        # Convert tasks to jax arrays (the Sigma and V)
        tasks_jax = [{"Sigma": jnp.asarray(t["Sigma"]),
                      "V": jnp.asarray(t["V"])} for t in tasks_np]
        sigmas_stack = jnp.stack([t["Sigma"] for t in tasks_jax])

        cond_results = {}
        for cond in CONDITIONS:
            # Split rng for each condition
            cond_rng, trial_rng = jax.random.split(trial_rng)
            res = run_trial(cond_rng, tasks_jax, cond)
            cond_results[cond.label] = res
        return cond_results, (sigmas_stack, U0)

    # Create a vectorised version over trials
    # We'll use a list comprehension for clarity (vmap over dicts less trivial)
    master_rng = jax.random.PRNGKey(0)
    trial_keys = jax.random.split(master_rng, N_TRIALS)

    all_trial_results = []
    all_sigmas = []
    for i, key in enumerate(trial_keys):
        cond_dict, (sigmas, U0) = run_all_conds_one_trial(key, i)
        all_trial_results.append(cond_dict)
        all_sigmas.append(sigmas)

    # ---- Aggregate across trials ----
    # For each condition, compute mean ft, etc.
    agg_results = {}
    for cond in CONDITIONS:
        ft_trials = jnp.stack([all_trial_results[i][cond.label]["ft"]
                               for i in range(N_TRIALS)])
        init_loss_trials = jnp.stack([all_trial_results[i][cond.label]["init_loss"]
                                      for i in range(N_TRIALS)])
        final_loss_trials = jnp.stack([all_trial_results[i][cond.label]["final_loss"]
                                       for i in range(N_TRIALS)])
        V_emp_trials = jnp.stack([all_trial_results[i][cond.label]["V_emp"]
                                  for i in range(N_TRIALS)])  # (trials, tasks, ...)

        agg_results[cond.label] = {
            "ft_mean": ft_trials.mean(axis=0),
            "ft_sem": ft_trials.std(axis=0) / jnp.sqrt(N_TRIALS),
            "init_loss_mean": init_loss_trials.mean(axis=0),
            "final_loss_mean": final_loss_trials.mean(axis=0),
            "V_emp_trials": V_emp_trials, 
            "V_emp_single": V_emp_trials[0],          # single trial for demo
        }

    # ---- Print summary ----
    print("\nForward transfer efficiency (mean ± SEM)")
    print(f"{'condition':14s}  ft_mean (all tasks)    ft_task_1")
    for cond in CONDITIONS:
        data = agg_results[cond.label]
        ft_all = data["ft_mean"]
        print(f"  {cond.label:14s}  {ft_all.mean():+.3f} ± {data['ft_sem'].mean():.3f}   "
              f"{ft_all[0]:+.3f}")

    # ---- Plots ----
    # Forward transfer curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_x = np.arange(2, N_TASKS + 1)
    for cond in CONDITIONS:
        data = agg_results[cond.label]
        ax.plot(task_x, data["ft_mean"], color=cond.color, ls=cond.linestyle, lw=1.8, label=cond.label)
        ax.fill_between(task_x, data["ft_mean"] - data["ft_sem"], data["ft_mean"] + data["ft_sem"],
                        color=cond.color, alpha=0.15)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_xlabel('task number')
    ax.set_ylabel(r'forward transfer efficiency  $1 - L_0^k / L_\mathrm{null}$')
    ax.set_title('Forward transfer across tasks (SGD, non‑linear network)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cl_sgd_ft.pdf", bbox_inches="tight")
    plt.close(fig)

    # Alignment matrix (averaged over trials and tasks)
    for cond in CONDITIONS:
        V_trials = agg_results[cond.label]["V_emp_trials"]  # (N_TRIALS, n_tasks, N_X, N_Y)
    
        # Vectorise the alignment computation across the independent trials
        trial_alignments = jax.vmap(compute_alignment_matrix)(V_trials)
        mean_alignment = trial_alignments.mean(axis=0)
        
        plot_alignment(mean_alignment, RESULTS_DIR / f"alignment_{cond.label.replace(' ','_')}.pdf")
        plot_individual_alignments(agg_results[cond.label]["V_emp_single"],
                                RESULTS_DIR / f"alignment_indiv_{cond.label.replace(' ','_')}.pdf")
    # Save raw data
    # data_out = {f"{cond.label.replace(' ','_')}_ft": agg_results[cond.label]["ft_mean"]
    #             for cond in CONDITIONS}
    # np.savez(RESULTS_DIR / "cl_sgd_data.npz", **data_out, s=S)

    print(f"\nPlots and data saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_2l_nl()