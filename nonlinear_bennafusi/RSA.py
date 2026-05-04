import jax
import jax.numpy as jnp
from functools import partial

def compare_representation_decompositions(U1: jnp.ndarray, U2: jnp.ndarray):
    """
    Compares two U vectors to return a matrix of similarity:
    M_ab = U1^{\top} @ U2
    Args:
        U1: singular vectors of shape PxP (column vectors)
        U2: singular vectors of shape PxP (column vectors)

    Returns:
        M: PxP similarity matrix between U1,U2
    """
    u1_shape, u2_shape = U1.shape, U2.shape
    assert u1_shape[0] == u1_shape[1], "U1 Not square"
    assert u1_shape[0] == u2_shape[0], "Dim 0 mismatch"
    assert u1_shape[1] == u2_shape[1], "Dim 1 mismatch"

    M = U1.T @ U2

    return M


@partial(jax.jit, static_argnames=['K', 'n_permutations'])
def cumul_subspace_overlap(U1: jnp.ndarray, U2: jnp.ndarray, K: int = 20, n_permutations: int = 2000):
    """
    Computes the overlap between truncated singular vectors.
    Measures how similar the leading k subspace is across tasks, normalized by k.
    Compares against randomized/shuffled k modes.
    
    Args: 
        U1: singular vectors of shape PxP (column vectors)
        U2: singular vectors of shape PxP (column vectors)
        K: Maximum number of modes to compute overlap for
        n_permutations: Number of random column shuffles for the null distribution

    Returns:
        overlaps_k: Vector of length K, actual subspace overlaps
        null_k_median: Median null overlap values for each k
        null_k_mean: Mean null overlap values for each k
        p_values: Empirical p-values for each k
        conf_bands: Tuple of arrays (lower_95, upper_95) for each k
    """
    P = U1.shape[1]
    K = min(K, P)
    
    # Establish a deterministic PRNGKey for the permutation generation inside the JIT scope
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, n_permutations)
    permuted_indices = jax.vmap(lambda rng: jax.random.permutation(rng, P))(keys)
    
    def calculate_for_k(k):
        # Base overlap computation
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        overlap = jnp.sum((U1_k.T @ U2_k)**2) / k
        
        # Null overlap distribution
        def single_null(idx):
            U1_null = U1[:, idx[:k]]
            return jnp.sum((U1_null.T @ U2_k)**2) / k
            
        nulls = jax.vmap(single_null)(permuted_indices)
        
        # Distribution analytics
        null_median = jnp.median(nulls)
        null_mean = jnp.mean(nulls)
        p_val = jnp.mean(nulls >= overlap)
        conf_lower = jnp.percentile(nulls, 2.5)
        conf_upper = jnp.percentile(nulls, 97.5)
        
        return overlap, null_median, null_mean, p_val, conf_lower, conf_upper

    # Statically unroll loop for K to circumvent JAX dynamic slicing constraints
    results = [calculate_for_k(k) for k in range(1, K + 1)]
    
    overlaps_k = jnp.array([r[0] for r in results])
    null_k_median = jnp.array([r[1] for r in results])
    null_k_mean = jnp.array([r[2] for r in results])
    p_values = jnp.array([r[3] for r in results])
    conf_bands = (jnp.array([r[4] for r in results]), jnp.array([r[5] for r in results]))
    
    return overlaps_k, null_k_median, null_k_mean, p_values, conf_bands


@jax.jit
def compute_representation_decomposition(h: jnp.ndarray):
    """
    Computes the representation covariance matrix R and its SVD decomposition.
    
    Args:
        h: Hidden representations matrix of shape (P, D_h).
        
    Returns:
        R: Covariance/Gram matrix of shape (P, P).
        lambdas: Singular values (eigenvalues) of shape (P,).
        U: Left singular vectors (eigenvectors) of shape (P, P). Columns are orthogonal
    """
    h_mean = jnp.mean(h, axis=0)
    h_centered = h - h_mean
    
    R = h_centered @ h_centered.T
    U, lambdas, _ = jnp.linalg.svd(R, full_matrices=False)
    
    return R, lambdas, U


def test_pipeline():
    """
    Tests the representation decomposition pipeline and subspace overlaps 
    with two distinct sets of random vectors.
    """
    key = jax.random.PRNGKey(42)
    key1, key2 = jax.random.split(key)
    
    P, D_h = 100, 64
    
    h1 = jax.random.normal(key1, (P, D_h))
    h2 = jax.random.normal(key2, (P, D_h))
    
    datasets = [("Dataset 1", h1), ("Dataset 2", h2)]
    U_matrices = []
    
    for name, h in datasets:
        R, lambdas, U = compute_representation_decomposition(h)
        U_matrices.append(U)
        
        R_reconstructed = (U * lambdas) @ U.T
        max_error = jnp.max(jnp.abs(R - R_reconstructed))
        
        print(f"--- {name} ---")
        print(f"Input shape (P, D_h): {h.shape}")
        print(f"R shape: {R.shape}")
        print(f"Reconstruction Error (Max Abs Diff): {max_error:.6e}\n")

    U1, U2 = U_matrices
    
    print("--- Representation Comparison ---")
    M = compare_representation_decompositions(U1, U2)
    print(f"Similarity Matrix M shape: {M.shape}")
    
    print("\n--- Subspace Overlap (K=20) ---")
    K_test = 20
    overlaps, null_med, null_mean, p_vals, conf_bands = cumul_subspace_overlap(
        U1, U2, K=K_test, n_permutations=2000
    )
    
    print(f"Overlaps shape: {overlaps.shape}")
    print(f"Null Medians shape: {null_med.shape}")
    print(f"P-values shape: {p_vals.shape}")
    print(f"Confidence Bands (Lower/Upper) shapes: {conf_bands[0].shape}, {conf_bands[1].shape}")
    
    # Print statistics for K=1 and K=20
    print(f"\nResults for k=1: Overlap = {overlaps[0]:.4f}, Null Median = {null_med[0]:.4f}, P-value = {p_vals[0]:.4f}")
    print(f"Results for k=20: Overlap = {overlaps[-1]:.4f}, Null Median = {null_med[-1]:.4f}, P-value = {p_vals[-1]:.4f}")


if __name__ == "__main__":
    test_pipeline()