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
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q, key

def make_spectrum(n):
    return jnp.exp(-2.0 * jnp.arange(n) / n)

def build_task_ensemble(key, n_tasks, V0_ext, s, M=M_EXT):
    K = len(s)
    s2 = _hybrid_variance_matrix(M)
    key_U, key = jax.random.split(key)
    U0, _ = _random_orthogonal(N_Y, key_U)

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
    W1, W2 = params
    return jax.nn.relu(x @ W1.T) @ W2.T

def loss_fn(params, x, y):
    pred = predict(params, x)
    return 0.5 * jnp.mean((y - pred) ** 2)

def _bf_coeffs(m, alpha):
    if m == 1:
        return jnp.zeros(0), 0.0
    k = jnp.arange(m - 1)
    g = alpha * (2.0 ** (-2 * k - 1))
    c_leak = alpha * (2.0 ** (-2 * m + 1))
    return g, c_leak

def bf_update(state, grads, lr, coeffs):
    W1, W2 = state
    m = W1.shape[0]

    if m == 1:
        return (W1.at[0].add(-lr * grads[0]), W2.at[0].add(-lr * grads[1]))

    g, c_leak = coeffs

    flux_W1 = g[:, None, None] * (W1[:-1] - W1[1:])
    flux_W2 = g[:, None, None] * (W2[:-1] - W2[1:])

    leak_W1 = c_leak * W1[-1:]
    leak_W2 = c_leak * W2[-1:]

    in_W1 = jnp.concatenate([-grads[0][None, ...], flux_W1], axis=0)
    out_W1 = jnp.concatenate([flux_W1, leak_W1], axis=0)
    dW1 = in_W1 - out_W1

    in_W2 = jnp.concatenate([-grads[1][None, ...], flux_W2], axis=0)
    out_W2 = jnp.concatenate([flux_W2, leak_W2], axis=0)
    dW2 = in_W2 - out_W2

    return (W1 + lr * dW1, W2 + lr * dW2)

# ----------------------------------------------------------------------
#  Task‑boundary reset
# ----------------------------------------------------------------------
def apply_reset(state, m, reset_spec):
    if reset_spec is None or reset_spec < 0:
        return state
    W1, W2 = state
    k = reset_spec
    W1 = W1.at[:k].set(W1[k])
    W2 = W2.at[:k].set(W2[k])
    return (W1, W2)

# ----------------------------------------------------------------------
#  Train one task
# ----------------------------------------------------------------------
def train_one_task(state, task_Sigma, key, m, coeffs, lr, n_iters, batch_size):
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
    
    key_test, key = jax.random.split(keys[-1])
    x_test = jax.random.normal(key_test, (TEST_BATCH, N_X))
    y_test = x_test @ task_Sigma.T
    final_loss = loss_fn((final_state[0][0], final_state[1][0]), x_test, y_test)
    return final_state, losses, final_loss, key

# ----------------------------------------------------------------------
#  Run full sequence
# ----------------------------------------------------------------------
def run_sequence(task_Sigmas, task_Vs, s, V0_obs, key, m, reset_spec, alpha=ALPHA_BF):
    n_tasks = len(task_Sigmas)
    
    key_init1, key_init2, key = jax.random.split(key, 3)
    init_W1 = jax.random.normal(key_init1, (N_H, N_X)) * jnp.sqrt(2.0 / N_X)
    init_W2 = jax.random.normal(key_init2, (N_Y, N_H)) * jnp.sqrt(2.0 / N_H)
    
    W1 = jnp.tile(init_W1, (m, 1, 1))
    W2 = jnp.tile(init_W2, (m, 1, 1))
    state = (W1, W2)

    coeffs = _bf_coeffs(m, alpha)

    init_losses = jnp.zeros(n_tasks)
    final_losses = jnp.zeros(n_tasks)
    null_losses = jnp.zeros(n_tasks)
    all_loss_curves = []
    selectivity = []

    for k in range(n_tasks):
        Sigma_k = task_Sigmas[k]
        if k > 0:
            state = apply_reset(state, m, reset_spec)

        null_losses = null_losses.at[k].set(0.5 * jnp.sum(Sigma_k ** 2))

        key_test, key = jax.random.split(key)
        x_test = jax.random.normal(key_test, (TEST_BATCH, N_X))
        y_test = x_test @ Sigma_k.T
        init_loss = loss_fn((state[0][0], state[1][0]), x_test, y_test)
        init_losses = init_losses.at[k].set(init_loss)

        key_train, key = jax.random.split(key)
        state, losses, final_loss, key = train_one_task(
            state, Sigma_k, key_train, m, coeffs, LR, N_ITERS, BATCH_SIZE
        )
        final_losses = final_losses.at[k].set(final_loss)
        all_loss_curves.append(losses)

        W0_eff = state[1][0] @ state[0][0]
        K_dim = V0_obs.shape[1]
        resp = jnp.array([jnp.linalg.norm(W0_eff @ V0_obs[:, a]) / s[a]
                          for a in range(K_dim)])
        selectivity.append(resp)

    ft = 1.0 - init_losses[1:] / null_losses[1:]

    return {
        "init_loss": init_losses,
        "final_loss": final_losses,
        "ft": ft,
        "loss_curves": jnp.stack(all_loss_curves),
        "selectivity": jnp.stack(selectivity),
        "V_matrices": task_Vs
    }

# ----------------------------------------------------------------------
#  Wrapper for vmap
# ----------------------------------------------------------------------
def single_trial(key, m, reset_spec):
    key_v0, key = jax.random.split(key)
    V0_ext, _ = jnp.linalg.qr(jax.random.normal(key_v0, (N_X, M_EXT)))
    V0_obs = V0_ext[:, :N_Y]
    s = make_spectrum(N_Y)

    (task_Sigmas, task_Vs, U0, key) = build_task_ensemble(
        key, N_TASKS, V0_ext, s, M_EXT
    )
    return run_sequence(task_Sigmas, task_Vs, s, V0_obs, key, m, reset_spec)

@partial(jax.jit, static_argnames=['m', 'reset_spec'])
def run_trials_vmap(keys, m, reset_spec):
    return jax.vmap(partial(single_trial, m=m, reset_spec=reset_spec))(keys)

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
#  Main experiment
# ----------------------------------------------------------------------
def run_2l_bf_nl():
    master_key = jax.random.PRNGKey(42)
    trial_keys = jax.random.split(master_key, N_TRIALS)

    all_results = {}
    for cond_idx, (cond_name, m, reset_spec) in enumerate(conditions):
        print(f"Running condition: {cond_name}")
        cond_keys = jax.vmap(lambda k: jax.random.fold_in(k, cond_idx))(trial_keys)
        results = run_trials_vmap(cond_keys, m=m, reset_spec=reset_spec)
        all_results[cond_name] = results
        print(f"  ft mean: {results['ft'].mean():.3f}")

    # ------ Analysis & Plots ------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_x = jnp.arange(2, N_TASKS+1)
    colors = ["C7", "C2", "C0", "C3", "C1", "k"]
    lss    = [":", "--", "-", "-", "-.", ":"]
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        ft_all = all_results[cond_name]["ft"]
        mean_ft = ft_all.mean(axis=0)
        sem_ft = ft_all.std(axis=0) / jnp.sqrt(N_TRIALS)
        ax.plot(task_x, mean_ft, color=col, ls=ls, label=cond_name)
        ax.fill_between(task_x, mean_ft - sem_ft, mean_ft + sem_ft, color=col, alpha=0.15)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set(xlabel="Task number", ylabel="Forward transfer efficiency",
           title=f"Forward transfer (mean ± SEM over {N_TRIALS} trials)")
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bf_sgd_ft.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(6,4))
    task_idx = 14
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        lc = all_results[cond_name]["loss_curves"]
        mean_lc = lc[:, task_idx, :].mean(axis=0)
        ax.plot(mean_lc, color=col, ls=ls, label=cond_name)
    ax.set_yscale('log')
    ax.set(xlabel="SGD step", ylabel="MSE loss",
           title=f"Learning curves on task {task_idx+1}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bf_sgd_lc.pdf")
    plt.close()

    example_V = all_results["no reset"]["V_matrices"][0]
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

    fig, ax = plt.subplots()
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        sel = all_results[cond_name]["selectivity"]
        lead = sel[:, :, :4].mean(axis=-1)
        trail = sel[:, :, 4:].mean(axis=-1) + 1e-12
        ratio = lead / trail
        ratio_mean = ratio.mean(axis=0)
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