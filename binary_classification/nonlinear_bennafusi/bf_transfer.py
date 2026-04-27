"""
Multi‑task continual learning with Benna‑Fusi synapses on a 2‑layer non‑linear network.
SGD training, JAX + vmap for efficiency.

Conditions:
  no_reset   — BF m=6, no reset
  reset_lv4  — BF m=6, reset levels 0..2 from level 3
  reset_lv5  — BF m=6, reset levels 0..3 from level 4
  reset_lv6  — BF m=6, reset levels 0..4 from level 5
  m4_lv4     — BF m=4, reset levels 0..2 from level 3
  no_BF      — m=1, no chain (forward‑transfer ceiling)
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from pathlib import Path

# ----------------------------------------------------------------------
#  Parameters
# ----------------------------------------------------------------------
N_Y = 10                # output / mode dimension
N_X = 30                # input dimension (>= M_EXT)
N_H = 10                # hidden layer size
M_EXT = 30              # extended rotation dimension

ALPHA_BF = 0.25          # BF coupling strength
LR = 0.05                # SGD learning rate
BATCH_SIZE = 256
N_ITERS = 2000           # SGD steps per task
N_TASKS = 20
N_TRIALS = 10            # independent task sequences
TEST_BATCH = 2000        # for evaluating loss after training

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
#  JAX‑ified task ensemble (fixed U₀, same hybrid variance)
# ----------------------------------------------------------------------
def _hybrid_variance_matrix(M):
    sigma_0, p, sigma_band, bw, alpha_0 = 1.0, 1.0, 1.0, 1.0, 0.0
    alpha = jnp.arange(M, dtype=jnp.float32)
    g = ((alpha + 1 + alpha_0) / (M + alpha_0)) ** p
    s2 = sigma_0 ** 2 * g[:, None] * g[None, :]
    diff = alpha[:, None] - alpha[None, :]
    s2 += sigma_band ** 2 * jnp.exp(-diff ** 2 / (2 * bw ** 2))
    s2 = s2.at[jnp.diag_indices(M)].set(0.0)
    return s2

def _sample_rotation(s2, key):
    M = s2.shape[0]
    std = jnp.sqrt(s2)
    key, subkey = jax.random.split(key)
    A = jnp.zeros((M, M))
    iu = jnp.triu_indices(M, k=1)
    A = A.at[iu].set(std[iu] * jax.random.normal(subkey, iu[0].shape))
    A = A - A.T
    Q = jax.scipy.linalg.expm(A)
    return Q, key

def _random_orthogonal(n, key):
    key, subkey = jax.random.split(key)
    H = jax.random.normal(subkey, (n, n))
    Q, R = jnp.linalg.qr(H)
    # ensure Haar measure
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q, key

def make_spectrum(n):
    return jnp.exp(-2.0 * jnp.arange(n) / n)

def build_task_ensemble(key, n_tasks, V0_ext, s, M=M_EXT):
    """Generate n_tasks tasks with fixed U0 and shared mode subspace."""
    K = len(s)
    s2 = _hybrid_variance_matrix(M)
    key_U, key = jax.random.split(key)
    U0, _ = _random_orthogonal(N_Y, key_U)   # same U for all tasks

    tasks_Sigma = []
    tasks_V = []
    for _ in range(n_tasks):
        key, subkey = jax.random.split(key)
        Q, subkey = _sample_rotation(s2, subkey)
        Vk = V0_ext @ Q[:, :K]
        Sigma = U0 @ jnp.diag(s) @ Vk.T
        tasks_Sigma.append(Sigma)
        tasks_V.append(Vk)

    return jnp.stack(tasks_Sigma), jnp.stack(tasks_V), U0, key


# ----------------------------------------------------------------------
#  Network and BF update
# ----------------------------------------------------------------------
def predict(params, x):
    """params = (W1, W2) for level 0 (only active weights)."""
    W1, W2 = params
    return jax.nn.relu(x @ W1.T) @ W2.T

def loss_fn(params, x, y):
    pred = predict(params, x)
    return 0.5 * jnp.mean((y - pred) ** 2)

def _bf_coeffs(m, alpha):
    """Precompute coupling coefficients (same as original ODE)."""
    if m == 1:
        return (0.,)*5   # no coupling
    c_fast = 0.5 * alpha
    mi_mid = jnp.arange(2, m)                # 2..m-1 (math indices)
    cl_mid = 2.0 ** (-2*mi_mid + 2) * alpha
    cr_mid = 2.0 ** (-2*mi_mid + 1) * alpha
    cl_slow = 2.0 ** (-2*m + 2) * alpha
    c_leak  = 2.0 ** (-2*m + 1) * alpha
    return c_fast, cl_mid, cr_mid, cl_slow, c_leak

def bf_update(state, grads, lr, coeffs):
    """One SGD + BF relaxation step.
    state: (W1_all, W2_all) where W1_all shape (m, N_H, N_X), W2_all (m, N_Y, N_H)
    grads: (grad_W1_0, grad_W2_0) for level 0 only.
    """
    W1_all, W2_all = state
    m = W1_all.shape[0]
    c_fast, cl_mid, cr_mid, cl_slow, c_leak = coeffs

    # Update level 0 with gradient + fast coupling (if m>1)
    dW1_0 = -grads[0]
    dW2_0 = -grads[1]
    if m > 1:
        dW1_0 += -c_fast * (W1_all[0] - W1_all[1])
        dW2_0 += -c_fast * (W2_all[0] - W2_all[1])
    new_W1_0 = W1_all[0] + lr * dW1_0
    new_W2_0 = W2_all[0] + lr * dW2_0

    # Update deeper levels (coupling only)
    new_W1 = W1_all.at[1:].get()   # avoid in-place
    new_W2 = W2_all.at[1:].get()
    if m > 2:
        # levels 1 .. m-2  (0‑indexed 1..m-2)
        w1_mid = W1_all[1:-1]   # (mid_len, N_H, N_X)
        w2_mid = W2_all[1:-1]
        # cl_mid, cr_mid shape (mid_len,)
        # dW_i = cl_mid*(W_{i-1} - W_i) - cr_mid*(W_i - W_{i+1})
        dW1_mid = cl_mid[:,None,None]*(W1_all[:-2] - w1_mid) \
                  - cr_mid[:,None,None]*(w1_mid - W1_all[2:])
        dW2_mid = cl_mid[:,None,None]*(W2_all[:-2] - w2_mid) \
                  - cr_mid[:,None,None]*(w2_mid - W2_all[2:])
        new_W1 = new_W1.at[:-1].add(lr * dW1_mid)
        new_W2 = new_W2.at[:-1].add(lr * dW2_mid)

    # Slowest level (m-1)
    dW1_slow = cl_slow*(W1_all[-2] - W1_all[-1]) - c_leak*W1_all[-1]
    dW2_slow = cl_slow*(W2_all[-2] - W2_all[-1]) - c_leak*W2_all[-1]
    new_W1 = new_W1.at[-1].add(lr * dW1_slow)
    new_W2 = new_W2.at[-1].add(lr * dW2_slow)

    # Put level 0 back
    new_W1 = new_W1.at[0].set(new_W1_0)
    new_W2 = new_W2.at[0].set(new_W2_0)

    return (new_W1, new_W2)

# ----------------------------------------------------------------------
#  Task‑boundary reset (JAX version)
# ----------------------------------------------------------------------
def apply_reset(state, m, reset_spec):
    if reset_spec is None:
        return state
    W1, W2 = state
    if isinstance(reset_spec, int):
        k = reset_spec
        W1 = W1.at[:k].set(W1[k])
        W2 = W2.at[:k].set(W2[k])
    else:   # array of weights
        w = jnp.asarray(reset_spec, dtype=W1.dtype) / jnp.sum(reset_spec)
        new_W1_0 = sum(w[i] * W1[i] for i in range(m))
        new_W2_0 = sum(w[i] * W2[i] for i in range(m))
        W1 = W1.at[0].set(new_W1_0)
        W2 = W2.at[0].set(new_W2_0)
    return (W1, W2)

# ----------------------------------------------------------------------
#  Train one task (return final state, loss curve, final loss)
# ----------------------------------------------------------------------
def train_one_task(state, task_Sigma, key, m, coeffs, lr, n_iters, batch_size):
    # We will record loss on a large test set at the end.
    # For learning curves we can periodically evaluate on a fixed test batch.
    def step(state, step_key):
        key_batch, key = jax.random.split(step_key)
        x = jax.random.normal(key_batch, (batch_size, N_X))
        y = x @ task_Sigma.T
        active_params = (state[0][0], state[1][0])
        loss, grads = jax.value_and_grad(loss_fn)(active_params, x, y)
        new_state = bf_update(state, grads, lr, coeffs)
        return new_state, loss

    keys = jax.random.split(key, n_iters)
    final_state, losses = jax.lax.scan(step, state, keys)
    # Final population loss (large test set)
    key_test, key = jax.random.split(keys[-1])
    x_test = jax.random.normal(key_test, (TEST_BATCH, N_X))
    y_test = x_test @ task_Sigma.T
    final_loss = loss_fn((final_state[0][0], final_state[1][0]), x_test, y_test)
    return final_state, losses, final_loss, key

# ----------------------------------------------------------------------
#  Run full sequence (one trial, one condition)
# ----------------------------------------------------------------------
def run_sequence(task_Sigmas, task_Vs, s, V0_obs, key, m, reset_spec, alpha=ALPHA_BF):
    """Takes pre‑generated task Sigmas/Vs and runs the task‑by‑task loop."""
    n_tasks = len(task_Sigmas)
    # Initialise state
    key_init, key = jax.random.split(key)
    W1 = jnp.zeros((m, N_H, N_X))
    W2 = jnp.zeros((m, N_Y, N_H))
    W1 = W1.at[0].set(1e-2 * jax.random.normal(key_init, (N_H, N_X)))
    W2 = W2.at[0].set(1e-2 * jax.random.normal(key_init, (N_Y, N_H)))
    state = (W1, W2)

    coeffs = _bf_coeffs(m, alpha)

    # Accumulators
    init_losses = jnp.zeros(n_tasks)
    final_losses = jnp.zeros(n_tasks)
    null_losses = jnp.zeros(n_tasks)
    all_loss_curves = []   # list of loss curves per task
    selectivity = []       # V0 response per task

    for k in range(n_tasks):
        Sigma_k = task_Sigmas[k]
        # task‑boundary reset (skip first task)
        if k > 0:
            state = apply_reset(state, m, reset_spec)

        # Null loss (weights zero)
        null_loss = 0.5 * jnp.sum(Sigma_k ** 2)
        null_losses = null_losses.at[k].set(null_loss)

        # Init loss (before training)
        init_loss = loss_fn((state[0][0], state[1][0]),
                            jnp.zeros((1, N_X)), jnp.zeros((1, N_Y)))  # dummy just using active params
        # Better: compute on a test batch
        key_test, key = jax.random.split(key)
        x_test = jax.random.normal(key_test, (TEST_BATCH, N_X))
        y_test = x_test @ Sigma_k.T
        init_loss = loss_fn((state[0][0], state[1][0]), x_test, y_test)
        init_losses = init_losses.at[k].set(init_loss)

        # Train
        key_train, key = jax.random.split(key)
        state, losses, final_loss, key = train_one_task(
            state, Sigma_k, key_train, m, coeffs, LR, N_ITERS, BATCH_SIZE
        )
        final_losses = final_losses.at[k].set(final_loss)
        all_loss_curves.append(losses)

        # Selectivity: compute linear product W2[0]@W1[0] response to V0 modes
        W0_eff = state[1][0] @ state[0][0]   # (N_Y, N_X)
        K = V0_obs.shape[1]
        resp = jnp.array([jnp.linalg.norm(W0_eff @ V0_obs[:, a]) / s[a]
                          for a in range(K)])
        selectivity.append(resp)

    # Forward transfer efficiency (task 2..N)
    ft = 1.0 - init_losses[1:] / null_losses[1:]

    return {
        "init_loss": init_losses,
        "final_loss": final_losses,
        "ft": ft,
        "loss_curves": jnp.stack(all_loss_curves),   # (n_tasks, n_iters)
        "selectivity": jnp.stack(selectivity),        # (n_tasks, K)
        "V_matrices": task_Vs                         # (n_tasks, N_X, K)
    }


# ----------------------------------------------------------------------
#  Wrapper for vmap (one trial)
# ----------------------------------------------------------------------
def single_trial(key, m, reset_spec):
    """Generate tasks and run a full trial. Returns all metrics."""
    # Build extended basis and spectrum (shared across trials)
    key_v0, key = jax.random.split(key)
    V0_ext, _ = jnp.linalg.qr(jax.random.normal(key_v0, (N_X, M_EXT)))
    V0_obs = V0_ext[:, :N_Y]
    s = make_spectrum(N_Y)

    # Generate task ensemble
    (task_Sigmas, task_Vs, U0, key) = build_task_ensemble(
        key, N_TASKS, V0_ext, s, M_EXT
    )
    del U0   # not needed further

    results = run_sequence(task_Sigmas, task_Vs, s, V0_obs, key, m, reset_spec)
    return results

# ----------------------------------------------------------------------
#  Conditions
# ----------------------------------------------------------------------
conditions = [
    ("no reset",  6, None),
    ("reset lv4", 6, 3),
    ("reset lv5", 6, 4),
    ("reset lv6", 6, 5),
    ("m=4 lv4",   4, 3),
    ("no-BF",     1, None),
]

# ----------------------------------------------------------------------
#  Main experiment with vmap
# ----------------------------------------------------------------------
def run_2l_bf_nl():
    # Prepare PRNG keys for all trials and conditions
    master_key = jax.random.PRNGKey(42)
    trial_keys = jax.random.split(master_key, N_TRIALS)

    # Create a grid: for each trial and each condition, we need a unique key.
    # We'll vmap over (trial_key, condition_index).
    # So we need to expand keys and cond indices to same shape.
    n_cond = len(conditions)

    # keys: (N_TRIALS, n_cond, 2) maybe just split each trial key further
    trial_keys_rep = jnp.repeat(trial_keys[:, None], n_cond, axis=1)  # (N_TRIALS, n_cond)
    # condition indices: (N_TRIALS, n_cond)
    cond_idx = jnp.arange(n_cond)[None, :] + jnp.zeros((N_TRIALS, n_cond), dtype=int)

    # Build static parameters for each condition
    m_list = jnp.array([c[1] for c in conditions])
    reset_list = jnp.array([c[2] if isinstance(c[2], int) else -1 for c in conditions], dtype=int)  # placeholder for None

    # vmap over leading two axes
    @partial(jax.vmap, in_axes=(0, 0, 0, None))
    def vmapped_run(key, m_val, reset_val, unused_cond_idx):
        # We need to treat reset_spec: None cannot be in JAX array directly.
        # So we pass it via a separate static list.
        # Better: use jax.vmap with static_argnums? We'll do a loop over conditions and vmap over trials only.
        pass

    # Instead, we loop over conditions and vmap over trials.
    all_results = {}
    for cond_idx, (cond_name, m, reset_spec) in enumerate(conditions):
        print(f"Running condition: {cond_name}")
        # For each trial, create a fresh key from trial_keys
        # We'll split each trial_key into a dedicated key for this condition
        cond_keys = jax.vmap(lambda k: jax.random.fold_in(k, cond_idx))(trial_keys)
        # vmap over trials
        run_fn = partial(single_trial, m=m, reset_spec=reset_spec)
        vmapped = jax.vmap(run_fn)
        results = vmapped(cond_keys)   # dict of arrays with leading axis N_TRIALS
        all_results[cond_name] = results
        print(f"  ft mean: {results['ft'].mean():.3f}")

    # ------ Analysis & Plots ------
    # 1. Forward transfer efficiency
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_x = jnp.arange(2, N_TASKS+1)
    colors = ["C7", "C2", "C0", "C3", "C1", "k"]
    lss    = [":", "--", "-", "-", "-.", ":"]
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        ft_all = all_results[cond_name]["ft"]   # (N_TRIALS, N_TASKS-1)
        mean_ft = ft_all.mean(axis=0)
        sem_ft = ft_all.std(axis=0) / jnp.sqrt(N_TRIALS)
        ax.plot(task_x, mean_ft, color=col, ls=ls, label=cond_name)
        ax.fill_between(task_x, mean_ft - sem_ft, mean_ft + sem_ft,
                        color=col, alpha=0.15)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set(xlabel="Task number", ylabel="Forward transfer efficiency",
           title=f"Forward transfer (mean ± SEM over {N_TRIALS} trials)")
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bf_sgd_ft.pdf")
    plt.close()

    # 2. Learning curves (task 15)
    fig, ax = plt.subplots(figsize=(6,4))
    task_idx = 14
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        lc = all_results[cond_name]["loss_curves"]   # (N_TRIALS, N_TASKS, N_ITERS)
        mean_lc = lc[:, task_idx, :].mean(axis=0)    # average over trials
        ax.plot(mean_lc, color=col, ls=ls, label=cond_name)
    ax.set_yscale('log')
    ax.set(xlabel="SGD step", ylabel="MSE loss",
           title=f"Learning curves on task {task_idx+1}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bf_sgd_lc.pdf")
    plt.close()

    # 3. Alignment matrices (using true V from tasks)
    # Assume the V matrices are identical across trials? They differ per trial because task rotations are random.
    # We'll pick the first trial for demonstration, or average over trials.
    # The provided functions compute alignment from V_emp (n_tasks, N_X, K). We'll use one trial.
    example_V = all_results["no reset"]["V_matrices"][0]  # (N_TASKS, N_X, N_Y)
    def compute_alignment_matrix(V_emp):
        num_tasks, *_ = V_emp.shape
        dots = jnp.abs(jnp.einsum('kai, laj -> klij', V_emp, V_emp))
        mask = 1.0 - jnp.eye(num_tasks)
        alignment = jnp.einsum('klab, kl -> ab', dots, mask) / (num_tasks * (num_tasks - 1))
        return alignment

    alignment = compute_alignment_matrix(example_V)
    plt.figure(figsize=(8,6))
    plt.imshow(alignment, cmap='viridis', origin='upper')
    plt.colorbar(label='Average Absolute Dot Product')
    plt.xlabel(r'Mode $\beta$ (Task $k^{\prime}$)')
    plt.ylabel(r'Mode $\alpha$ (Task $k$)')
    plt.title(r'Cross‑Task Mode Alignment $\langle v_\alpha^{(k)}, v_\beta^{(k^\prime)} \rangle$')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mode_alignment_avg.pdf")
    plt.close()

    # Pairwise alignment heatmaps
    n_tasks_show = min(5, N_TASKS)
    fig, axes = plt.subplots(n_tasks_show, n_tasks_show,
                             figsize=(2*n_tasks_show, 2*n_tasks_show))
    for i in range(n_tasks_show):
        for j in range(n_tasks_show):
            if j > i:
                axes[i][j].axis('off')
                continue
            M_ab = jnp.abs(example_V[i] @ example_V[j].T)
            im = axes[i][j].imshow(M_ab, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            axes[i][j].set_title(f"M  T:{i},{j}")
            if i == n_tasks_show-1:
                axes[i][j].set_xlabel(r'Mode $\beta$')
            if j == 0:
                axes[i][j].set_ylabel(r'Mode $\alpha$')
    plt.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Absolute Magnitude')
    plt.savefig(RESULTS_DIR / "mode_alignment_pairs.pdf")
    plt.close()

    # 4. Selectivity ratio (mean over tasks)
    fig, ax = plt.subplots()
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        sel = all_results[cond_name]["selectivity"]  # (N_TRIALS, N_TASKS, K)
        # ratio lead/trail: modes 1-4 vs 5-10
        lead = sel[:, :, :4].mean(axis=-1)
        trail = sel[:, :, 4:].mean(axis=-1) + 1e-12
        ratio = lead / trail
        ratio_mean = ratio.mean(axis=0)   # over trials
        ratio_sem = ratio.std(axis=0) / jnp.sqrt(N_TRIALS)
        ax.errorbar(jnp.arange(1, N_TASKS+1), ratio_mean, yerr=ratio_sem,
                     color=col, ls=ls, label=cond_name)
    ax.axhline(1.0, color='k', ls=':', alpha=0.4)
    ax.set(xlabel='Task number', ylabel='Lead / Trail V0 response ratio',
           title='Mode selectivity (linear W2W1)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bf_sgd_selectivity.pdf")
    plt.close()

    print(f"\nAll figures saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_2l_bf_nl()