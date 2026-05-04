import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from pathlib import Path

# ----------------------------------------------------------------------
#  Parameters
# ----------------------------------------------------------------------
N_Y = 10                
N_X = 30                
N_H = 10                
M_EXT = 30              

ALPHA_BF = 0.25          
LR = 0.05                
BATCH_SIZE = 256
N_ITERS = 5000           
N_TASKS = 10
N_TRIALS = 10  # repeats          
TEST_BATCH = 2000        

max_pow = int(np.log2(N_ITERS)) if N_ITERS > 0 else 0
_cp = [0] + [2**i for i in range(max_pow + 1)]
if _cp[-1] < N_ITERS - 1:
    _cp.append(N_ITERS - 1)
CHECKPOINTS = jnp.unique(jnp.array(_cp, dtype=jnp.int32))
NUM_CP = len(CHECKPOINTS) # perform checkpointing to as to reduce computation

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
#  JAX‑ified task ensemble
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
    return 0.5 * jnp.mean(jnp.sum((y - pred) ** 2, axis=-1))

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

def apply_reset(state, m, reset_spec):
    if reset_spec is None or reset_spec < 0:
        return state
    W1, W2 = state
    k = reset_spec
    W1 = W1.at[:k].set(W1[k])
    W2 = W2.at[:k].set(W2[k])
    return (W1, W2)

def train_one_task(state, task_Sigma, key, m, coeffs, lr, n_iters, batch_size):
    def step(state, step_key):
        key_batch, key = jax.random.split(step_key)
        x = jax.random.normal(key_batch, (batch_size, N_X))
        y = x @ task_Sigma.T
        active_params = (state[0][0], state[1][0])
        loss, grads = jax.value_and_grad(loss_fn)(active_params, x, y)
        new_state = bf_update(state, grads, lr, coeffs)
        return new_state, (new_state, loss)

    keys = jax.random.split(key, n_iters)
    final_state, (history_states, losses) = jax.lax.scan(step, state, keys)
    
    all_W1 = jnp.concatenate([state[0][None, 0], history_states[0][:, 0]], axis=0)
    all_W2 = jnp.concatenate([state[1][None, 0], history_states[1][:, 0]], axis=0)
    cp_W1 = all_W1[CHECKPOINTS]
    cp_W2 = all_W2[CHECKPOINTS]

    key_test, key = jax.random.split(keys[-1])
    x_test = jax.random.normal(key_test, (TEST_BATCH, N_X))
    y_test = x_test @ task_Sigma.T
    final_loss = loss_fn((final_state[0][0], final_state[1][0]), x_test, y_test)
    return final_state, cp_W1, cp_W2, losses, final_loss, key

# ----------------------------------------------------------------------
#  Scanned Run Sequence (Optimized)
# ----------------------------------------------------------------------
def run_sequence(task_Sigmas, task_Vs, s, V0_obs, key, m, reset_spec, alpha=ALPHA_BF):
    n_tasks = len(task_Sigmas)
    
    key_init1, key_init2, key_test, key_trains = jax.random.split(key, 4)
    init_W1 = jax.random.normal(key_init1, (N_H, N_X)) * jnp.sqrt(2.0 / N_X)
    init_W2 = jax.random.normal(key_init2, (N_Y, N_H)) * jnp.sqrt(2.0 / N_H)
    
    W1 = jnp.tile(init_W1, (m, 1, 1))
    W2 = jnp.tile(init_W2, (m, 1, 1))
    state = (W1, W2)

    coeffs = _bf_coeffs(m, alpha)

    x_tests = jax.random.normal(key_test, (n_tasks, TEST_BATCH, N_X))
    y_tests = jax.vmap(lambda x, sig: x @ sig.T)(x_tests, task_Sigmas)
    train_keys = jax.random.split(key_trains, n_tasks)

    eval_fn_vmap = jax.vmap(
        jax.vmap(
            lambda w1, w2, x, y: loss_fn((w1, w2), x, y), 
            in_axes=(None, None, 0, 0)
        ), 
        in_axes=(0, 0, None, None)
    )

    rep_fn_vmap = jax.vmap(
        jax.vmap(
            lambda w1, x: jax.nn.relu(w1 @ x.T), 
            in_axes=(None, 0)
        ),
        in_axes=(0, None)
    )

    def task_step(carry, task_data):
        state, k = carry
        Sigma_k, V_k, x_test_k, y_test_k, tk_key = task_data

        state = jax.lax.cond(
            k > 0,
            lambda s: apply_reset(s, m, reset_spec),
            lambda s: s,
            state
        )

        null_loss = 0.5 * jnp.sum(Sigma_k ** 2)
        init_loss = loss_fn((state[0][0], state[1][0]), x_test_k, y_test_k)

        state, cp_W1, cp_W2, losses, final_loss, _ = train_one_task(
            state, Sigma_k, tk_key, m, coeffs, LR, N_ITERS, BATCH_SIZE
        )

        cp_losses = eval_fn_vmap(cp_W1, cp_W2, x_tests, y_tests)
        
        cp_reps = rep_fn_vmap(cp_W1, x_tests)

        W0_eff = state[1][0] @ state[0][0]
        resp = jnp.linalg.norm(W0_eff @ V0_obs, axis=0) / s

        out = {
            "init_loss": init_loss,
            "final_loss": final_loss,
            "null_loss": null_loss,
            "loss_curves": losses,
            "selectivity": resp,
            "cp_W1": cp_W1,
            "cp_W2": cp_W2,
            "cross_eval_losses": cp_losses,
            "cp_reps": cp_reps
        }
        return (state, k + 1), out

    task_data = (task_Sigmas, task_Vs, x_tests, y_tests, train_keys)
    _, history = jax.lax.scan(task_step, (state, 0), task_data)

    ft = 1.0 - history["init_loss"][1:] / history["null_loss"][1:]

    return {
        "init_loss": history["init_loss"],
        "final_loss": history["final_loss"],
        "ft": ft,
        "loss_curves": history["loss_curves"],
        "selectivity": history["selectivity"],
        "V_matrices": task_Vs,
        "cross_eval_losses": history["cross_eval_losses"],
        "cp_reps": history["cp_reps"]
    }

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
        
        results = jax.device_get(run_trials_vmap(cond_keys, m=m, reset_spec=reset_spec))
        all_results[cond_name] = results
        
        print(f"  ft mean: {results['ft'].mean():.3f}")

        safe_name = cond_name.replace(" ", "_").replace("=", "")
        cond_dir = RESULTS_DIR / safe_name
        cond_dir.mkdir(exist_ok=True, parents=True) 
        
        # results["cp_reps"] shape: (N_TRIALS, N_TASKS_TRAIN, NUM_CP, N_TASKS_EVAL, N_H, TEST_BATCH)
        for k in range(N_TASKS):
            task_dir = cond_dir / f"task_{k}"
            task_dir.mkdir(exist_ok=True)
            for k_prime in range(N_TASKS):
                # Shape extraction: (N_TRIALS, NUM_CP, N_H, TEST_BATCH)
                reps_k_kprime = results["cp_reps"][:, k, :, k_prime, :, :]
                np.save(task_dir / f"eval_task_{k_prime}.npy", reps_k_kprime)

    # ------ Analysis & Plots ------
    colors = ["C7", "C2", "C0", "C3", "C1", "k"]
    lss    = [":", "--", "-", "-", "-.", ":"]
    
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_x = np.arange(2, N_TASKS+1)
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        ft_all = all_results[cond_name]["ft"]
        mean_ft = ft_all.mean(axis=0)
        sem_ft = ft_all.std(axis=0) / np.sqrt(N_TRIALS)
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
    task_idx = int(N_TASKS/2)
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

    # fig, ax = plt.subplots()
    # for (cond_name, *_), col, ls in zip(conditions, colors, lss):
    #     sel = all_results[cond_name]["selectivity"]
    #     lead = sel[:, :, :4].mean(axis=-1)
    #     trail = sel[:, :, 4:].mean(axis=-1) + 1e-12
    #     ratio = lead / trail
    #     ratio_mean = ratio.mean(axis=0)
    #     ratio_sem = ratio.std(axis=0) / np.sqrt(N_TRIALS)
    #     ax.errorbar(np.arange(1, N_TASKS+1), ratio_mean, yerr=ratio_sem,
    #                  color=col, ls=ls, label=cond_name)
    # ax.axhline(1.0, color='k', ls=':', alpha=0.4)
    # ax.set(xlabel='Task number', ylabel='Lead / Trail V0 response ratio',
    #        title='Mode selectivity (linear W2W1)')
    # ax.legend()
    # fig.tight_layout()
    # fig.savefig(RESULTS_DIR / "bf_sgd_selectivity.pdf")
    # plt.close()

# ------ Learning Curve Matrix ------
    fig, axes = plt.subplots(N_TASKS, N_TASKS, figsize=(24, 24), sharex=True, sharey=True)
    legend_lines = []
    legend_labels = []

    for row_eval in range(N_TASKS):
        for col_train in range(N_TASKS):
            ax = axes[row_eval, col_train]
            if row_eval == N_TASKS - 1:
                ax.set_xlabel(f"Tr {col_train+1}")
            if col_train == 0:
                ax.set_ylabel(f"Ev {row_eval+1}")

            for (cond_name, *_), col, ls in zip(conditions, colors, lss):
                mean_cross = all_results[cond_name]["cross_eval_losses"].mean(axis=0)
                y = mean_cross[col_train, :, row_eval]
                line, = ax.plot(CHECKPOINTS, y, color=col, ls=ls)
                
                if row_eval == 0 and col_train == 0:
                    legend_lines.append(line)
                    legend_labels.append(cond_name)
            
            ax.set_xscale('symlog', linthresh=1.0)
            # ax.set_yscale('log')
            if col_train > 0: ax.tick_params(labelleft=False)
            if row_eval < N_TASKS - 1: ax.tick_params(labelbottom=False)

    fig.legend(legend_lines, legend_labels, loc='upper center', ncol=len(conditions), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97]) 
    fig.savefig(RESULTS_DIR / "learning_curve_matrix.pdf")
    plt.close()

    # ------ Retention Plot ------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    task_x = np.arange(2, N_TASKS + 1)
    
    for (cond_name, *_), col, ls in zip(conditions, colors, lss):
        # cross_eval_losses shape: (N_TRIALS, N_TASKS_TRAIN, NUM_CP, N_TASKS_EVAL)
        cross_eval = all_results[cond_name]["cross_eval_losses"]
        
        # Extract evaluation losses at the final checkpoint for each training task
        # Shape: (N_TRIALS, N_TASKS_TRAIN, N_TASKS_EVAL)
        final_losses = cross_eval[:, :, -1, :] 
        
        retention_trials = []
        for t_prime in range(1, N_TASKS):
            # Calculate retention for all tasks k learned prior to t_prime.
            # retention = loss_{k,k} - loss_{k, t_prime}
            # Average this across all k < t_prime.
            ret_t = np.mean([
                final_losses[:, k, k] - final_losses[:, t_prime, k] 
                for k in range(t_prime)
            ], axis=0)
            retention_trials.append(ret_t)
            
        retention_all = np.stack(retention_trials, axis=1)
        
        mean_ret = retention_all.mean(axis=0)
        sem_ret = retention_all.std(axis=0) / np.sqrt(N_TRIALS)
        
        ax.plot(task_x, mean_ret, color=col, ls=ls, label=cond_name)
        ax.fill_between(task_x, mean_ret - sem_ret, mean_ret + sem_ret, color=col, alpha=0.15)

    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set(xlabel="Task number", ylabel="Average Retention",
           title=f"Average Retention (mean ± SEM over {N_TRIALS} trials)")
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bf_sgd_retention.pdf")
    plt.close()

    print(f"\nAll figures and representations saved to {RESULTS_DIR}")

if __name__ == "__main__":
    run_2l_bf_nl()