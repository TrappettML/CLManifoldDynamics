from chex import params_product
import jax
import jax.numpy as jnp
from jaxopt import OSQP
import matplotlib.pyplot as plt
from itertools import permutations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from random import sample
from functools import partial
import pickle

# @jax.jit(static_argnames=('P', 'M', 'N', 'n_t', 'qp_solver'))
def run_glue_solver(key, data: jax.Array, P:int, M:int, N:int, n_t:int, qp_solver):
    m_key, t_key, y_key = jax.random.split(key, 3)
    A = jnp.eye(N)
    H = jnp.zeros((P*M,1))
    m_data = data
    assert m_data.shape[0] == P, "M_data wrong dim 0 size"
    assert m_data.shape[1] == M, "M_data wrong dim 1 size"
    assert m_data.shape[2] == N, "M_data wrong dim 2 size"
    all_ts = get_all_ts(t_key, N, n_t)
    all_ys = get_all_ys(y_key, P, n_t)
    P_indices = jnp.array(list(permutations(range(P), r=2)))
    # solve the qp equation
    anchor_points, Gs, Primals = sample_anchor_points(m_data, all_ts, all_ys, A, H, qp_solver) # (aps, Gs, primals), shapes: ((n_t, P,N),(n_t,P*M,N),(n_t,N,1))
    centers, center_gram = get_ap_centers(anchor_points)
    ap_axes, t_1ks, axes_gram = get_aps_axis_var(anchor_points, centers, all_ts)

    # manifold geometries
    capacity = (1/P * jnp.mean(jax.vmap(lambda s,t: (s@t.T).T @ jnp.linalg.pinv(s@s.T, hermitian=True) @ (s@t.T), in_axes=(0,0))(anchor_points, all_ts)))**(-1)
    dimension = (1/P * jnp.mean(jax.vmap(lambda t,g: t.T @ jnp.linalg.pinv(g, hermitian=True) @ t, in_axes=(0,0))(t_1ks, axes_gram)))
    indiv_dim = jnp.mean(jax.vmap(lambda t,g: t * (jnp.linalg.pinv(g, hermitian=True)@t), in_axes=(0,0))(t_1ks, axes_gram), axis=0)
    radius = get_radius(t_1ks, axes_gram, center_gram)
    indiv_rad = get_indiv_radius(t_1ks, axes_gram, center_gram)
    center_align = get_center_alignment(centers, P, P_indices)
    axis_align = get_axis_alignment(ap_axes, P, P_indices)
    center_axis_align = get_center_axis_alignment(centers, ap_axes, P, P_indices)
    approx_capacity = (1 + radius**(-2))/dimension
    glue_metrics = (capacity, dimension, radius, center_align, axis_align, center_axis_align, approx_capacity)
    plotting_inputs = (M, anchor_points, Gs, all_ys, all_ts, Primals, centers, ap_axes)
    single_p_metrics = (indiv_dim, indiv_rad)
    return glue_metrics, plotting_inputs, single_p_metrics


# def get_m_data(key, data: jax.Array, P: int, M, N) -> jax.Array:
#     """Sample from full data to get M data points
#     data shape: (P classes, num Points, N features)"""
#     m_keys = jax.random.split(key, P)
#     n_data_points = data.shape[1]
#     def sample_m(k):
#         return jax.random.choice(k, n_data_points, shape=(M,), replace=False)
#     m_data_axes = jax.vmap(sample_m)(m_keys)
#     m_data = jax.vmap(lambda d, indcs: d[indcs,:])(data, m_data_axes)
#     assert m_data.shape[0] == P, "M_data wrong dim 0 size"
#     assert m_data.shape[1] == M, "M_data wrong dim 1 size"
#     assert m_data.shape[2] == N, "M_data wrong dim 2 size"
#     return m_data


def get_all_ts(key, N: int, n_t: int):
    t_mu = jnp.zeros(N)
    t_sigma = jnp.eye(N)
    return jax.random.multivariate_normal(key, t_mu, t_sigma, shape=(n_t,))


def get_all_ys(key, P:int, n_t: int):
    """"inputs:
        key, P, n_t, y_type
        y_type: one of ['one_v_rest', 'entropy']"""
    potential_ys = jnp.full((n_t, P), -1)
    one_indices = jax.random.randint(key, shape=(n_t,), minval=0, maxval=P)
    row_indices = jnp.arange(n_t)
    potential_ys = potential_ys.at[row_indices, one_indices].set(1)
    return potential_ys

def get_radius(t_1ks, G_1ks, G_0):
    def rad_top(t,g_1):
        return t.T @ jnp.linalg.pinv(g_1+G_0, hermitian=True) @ t
    def rad_bot(t,g_1):
        return t.T @ jnp.linalg.pinv(g_1 + (g_1 @ jnp.linalg.pinv(G_0, hermitian=True) @ g_1), hermitian=True) @ t
    # for neural data chou2025a
    # top_values = jax.vmap(rad_top)(t_1ks, G_1ks)
    # bot_values = jax.vmap(rad_bot)(t_1ks, G_1ks)
    # mean_top = jnp.mean(top_values)
    # mean_bot = jnp.mean(bot_values)
    # radius = jnp.sqrt(mean_top/mean_bot)
    # for neural networks representations
    radius = jnp.sqrt(jnp.mean(jax.vmap(lambda t,g: rad_top(t,g)/rad_bot(t,g), in_axes=(0,0))(t_1ks, G_1ks)))
    return radius

def get_indiv_radius(t_1ks, G_1ks, G_0):
    def rad_top(t,g_1):
        return t.T * jnp.linalg.pinv(g_1+G_0, hermitian=True) @ t
    def rad_bot(t,g_1):
        return t.T * jnp.linalg.pinv(g_1 + (g_1 @ jnp.linalg.pinv(G_0, hermitian=True) @ g_1), hermitian=True) @ t
    # for neural data chou2025a
    # top_values = jax.vmap(rad_top)(t_1ks, G_1ks)
    # bot_values = jax.vmap(rad_bot)(t_1ks, G_1ks)
    # mean_top = jnp.mean(top_values, axis=0)
    # mean_bot = jnp.mean(bot_values, axis=0)
    # radius = jnp.sqrt(mean_top/mean_bot)
    # for neural networks representations
    radius = jnp.sqrt(jnp.mean(jax.vmap(lambda t,g: rad_top(t,g)/rad_bot(t,g), in_axes=(0,0))(t_1ks, G_1ks)))
    return radius


def get_center_alignment(axis_centers, P, P_indices):
    inner_f = lambda ac,pi: (ac[pi[0]].T @ ac[pi[1]])/(jnp.linalg.norm(ac[pi[0]]) * jnp.linalg.norm(ac[pi[1]]))
    center_alignment = 1/(P * (P -1)) * jnp.sum(
        jax.vmap(inner_f, 
                    in_axes=(None,0))(axis_centers, P_indices))
    return center_alignment


def get_axis_alignment(all_aps_axis, P, P_indices):
    # all_aps_axis shape: (n_t,P,N)
    axis_alignment = 1/(P*(P-1)) * jnp.sum(
        jax.vmap(
            lambda x, pid: axis_align_nt_sum(x[:,pid[0],:],x[:,pid[1],:]),
            in_axes=(None, 0))
            (all_aps_axis, P_indices)
            )
    return axis_alignment


def get_center_axis_alignment(aps_centers, aps_axes, P, P_indices):
    center_axis_alignment = 1/(P*(P-1)) * jnp.sum(
        jax.vmap(
            lambda x,y, pid: center_axis_align_nt_sum(x[pid[0]],y[:, pid[1], :]),
            in_axes=(None, None, 0))
            (aps_centers, aps_axes, P_indices)
            )
    return center_axis_alignment


def sample_anchor_points(m_data: jax.Array, all_t_ks: jax.Array, all_y_ks: jax.Array, A: jax.Array, H: jax.Array, qp_solver):
    """
    inputs:
        m_data (shape): (P,M,N)
        all_y_ks (shape): (n_t, P)
    Return: APs, Gs, primals
        all_anchor_points (shape): (n_t, P, N)
        all_Gs (shape): (n_t, P, M, N)
        all_primals (shape):  (n_t, P, 1)
    """
    A = A
    Q = -1 * all_t_ks
    H = H
    n_t = all_t_ks.shape[0]
    P = m_data.shape[0]
    M = m_data.shape[1]
    N = m_data.shape[2]
    all_G = jax.vmap(lambda x,y: x[:,None, None] * y, in_axes=(0,None))(all_y_ks, m_data).reshape((n_t, -1, N))
    sols = jax.vmap(lambda a, q, g, h: qp_solver.run(params_obj=(a,q[:,None]), params_eq=None, params_ineq=(g,h)), in_axes=(None, 0, 0, None))(A, Q, all_G, H)
    primals = sols.params.primal
    duals = sols.params.dual_ineq
    all_lambdas = duals.reshape((n_t,P,M))
    Gs_pm = all_G.reshape((n_t,P,M,N))
    all_anchor_points = jax.vmap(single_ap, in_axes=(0,0))(all_lambdas, Gs_pm)
    assert all_anchor_points.shape[0] == n_t, "all aps not matching n_t, axis=0"
    assert all_anchor_points.shape[1] == P, "aps not matching P shape, axis=1"
    assert all_anchor_points.shape[2] == N, "aps not matching N shape, axis=2"
    return all_anchor_points, all_G, primals


def single_ap(lam, gpm):
    return jax.vmap(safe_divide, in_axes=(0,0))(jnp.sum((lam.T * gpm.T).T, axis=1), jnp.sum(lam, axis=1)) # vmap over P


def sample_single_anchor_point(y_k: jax.Array, t_k: jax.Array, m_data: jax.Array, H, A) -> tuple[jax.Array, jax.Array, jax.Array]:
    "Single anchor point function, calclulate anchor points for t and y "
    "return primals, G and anchorpoints"
    qp = OSQP(tol=1e-4)
    N = m_data.shape[2]
    M = m_data.shape[1]
    P = m_data.shape[0]
    y_k = jnp.expand_dims(y_k, axis=1)
    t_k = jnp.expand_dims(t_k, axis=1)
    q_t = -1 * t_k
    # G = jnp.concatenate(jax.vmap(lambda a,b: a*b, in_axes=(0,0))(y_k, m_data),axis=0)
    G = (y_k[:, :, None] * m_data).reshape(-1, N) # (P,1,1) * (P,M,N) -> (P*M,N)
    sol = qp.run(params_obj=(A, q_t), params_eq=None, params_ineq=(G, H)).params
    primal = sol.primal
    z_duals = sol.dual_ineq
    G_p_x_m = G.reshape(P, M,-1)
    lambda_p_x_m = z_duals.squeeze().reshape(P, M)
    ap_top = jnp.sum((lambda_p_x_m.T * G_p_x_m.T).T, axis=1) # resulting shape: PxN
    ap_bottom = jnp.sum(lambda_p_x_m, axis=1) # resulting shape: P
    anchor_points = jax.vmap(safe_divide, in_axes=(0,0))(ap_top, ap_bottom) # resulting shape: PxN
    return (anchor_points, G, primal)


def get_ap_centers(anchor_points):
    """aps shape: (n_t, P, N)"""
    s_0 = jnp.mean(anchor_points, axis=0) # shapes (n_t,P,N) -> (P,N) # anchor centers
    G_0 = s_0 @ s_0.T # shape: (PxN) @ (PxN).T = (P,P) # center gramm matrix
    return s_0, G_0


def get_aps_axis_var(anchor_points, centers, all_t_ks):
    # s_1ks.shape = apshape: (n_t,P,N) x centers:(P,N) // vmapped over P, makes P first index, reshape
    s_1ks = jax.vmap(lambda x,y: x-y, in_axes=(1,0))(anchor_points, centers).swapaxes(1,0) # swaps to (n_t, P, N)
    # t_1ks.shape (n_t,P) = s_1ks shape: (n_t,P,N) x all_t_ks.shape: (n_t, N)
    t_1ks = jax.vmap(lambda x,y: x@y.T, in_axes=(0,0))(s_1ks, all_t_ks)
    # G_1k.shape (n_t, P, P) = s_1ks shape:(n_t,P,N) {(P,N)x(N,P)}
    G_1k = jax.vmap(lambda x: x@x.T, in_axes=(0))(s_1ks)
    return s_1ks, t_1ks, G_1k


def axis_align_nt_sum(s_ik, s_jk):
    # s_*ks shape: (n_t, N)
    top = lambda x,y: x@y.T # along N axis
    bot = lambda x,y: jnp.linalg.norm(x) * jnp.linalg.norm(y)
    frac = lambda x,y: top(x,y)/bot(x,y)
    return jnp.mean(jax.vmap(frac, in_axes=(0,0))(s_ik, s_jk))


def center_axis_align_nt_sum(s_0i, s_kj):
    # s_ks shape: (n_t, N)
    # s_0j shape: (1,N)
    top = lambda x,y: x@y.T # along N axis
    bot = lambda x,y: jnp.linalg.norm(x) * jnp.linalg.norm(y)
    frac = lambda x,y: top(x,y)/bot(x,y)
    return jnp.mean(jax.vmap(frac, in_axes=(None, 0))(s_0i, s_kj))


def safe_divide(numerator, denominator, fill_value=0.0):
    """
    Performs division using jax.lax.div, returning fill_value where denominator is 0.
    """
    # 1. Create a mask where the denominator is NOT zero
    # We assume standard float comparison; for integers, logic is identical.
    mask = jax.lax.ne(denominator, 0.0)
    
    # 2. Create a "safe" denominator. 
    # We replace 0 with 1 to avoid generating NaNs/Infs during the calculation.
    # This is safer for gradients than simply dividing and masking afterwards.
    safe_denominator = jax.lax.select(mask, denominator, jnp.ones_like(denominator))
    
    # 3. Perform the raw division using the requested lax primitive
    raw_div = jax.lax.div(numerator, safe_denominator)
    
    # 4. Select the division result where mask is True, otherwise use fill_value
    # Ensure fill_value matches the shape/type of the result
    safe_fill = jnp.full_like(raw_div, fill_value)
    
    return jax.lax.select(mask, raw_div, safe_fill)


def make_plots(M:int, 
               anchor_points: jax.Array, 
               Gs: jax.Array, 
               all_y_ks: jax.Array, 
               all_t_ks: jax.Array, 
               primals: jax.Array, 
               centers: jax.Array, 
               ap_axes: jax.Array,
                plots_path: str| None = None, 
                n_plots: int=3):
    # Make 3 cone plots, 3 columns 1 row
    # Make 1 anchor point plot, APs, centers and axis maybe with points
    # want logic to handle N= 2 or 3 and TODO: if more than do PCA to embed. 
    # Ensure directory exists
    n_t, P, N = anchor_points.shape

    if plots_path:
        base_dir = os.path.dirname(plots_path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir)

        html_filename = f"{plots_path}/manifold_plots.html"

    k = min(n_plots, n_t)
    n_t_sample_indcs = sample(range(n_t), k)
    js_str = 'cdn'
    figs = []
    for i in n_t_sample_indcs:
        title_str = f"Geometries for n_t: {i}; y_k: {np.array(all_y_ks[i])}"
        figs.append(plot_cone_geometry(P, 
                                Gs[i], 
                                M, 
                                all_t_ks[i],
                                primals[i],
                                anchor_points[i],
                                title=title_str))
        
    figs.append(make_manifold_plot(anchor_points, centers, ap_axes))
    if plots_path:
        for fig in figs:
            # html plot
            with open(html_filename, 'w') as f:
                f.write(fig.to_html(full_html=False, include_plotlyjs=js_str))
                js_str = False
            # png plot
            png_name = f"{plots_path}/manifold_plot_n_t{i}.png"
            fig.write_image(png_name)
    return figs


def make_manifold_plot(anchor_points: jax.Array, centers: jax.Array, ap_axes: jax.Array):
    """
    Plots anchor points, centers, and axes. 
    self.anchor_points shape: (n_t, P, N)
    self.ap_axes shape: (n_t, P, N)
    self.centers shape: (P, N)
    - n_t: number of anchor points per class
    - P: number of classes
    - N: dimension (2 or 3)
    """

    # 1. Detect Dimension and Classes
    # Shape is (n_t, P, N)
    n_t, P, N = anchor_points.shape

    # 2. Trace Factory (Smart Wrapper)
    def make_scatter(coords, **kwargs):
        # Ensure coords is at least 2D array (1, dim) or (n_t, dim)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
            
        common_args = dict(x=coords[:, 0], y=coords[:, 1], **kwargs)
        
        if N == 3:
            return go.Scatter3d(z=coords[:, 2], **common_args)
        else:
            return go.Scatter(**common_args)

    # 3. Initialize Figure
    fig = go.Figure()
    
    # Color palette to match P classes
    colors = px.colors.qualitative.Plotly
    
    # 4. Loop over classes (Iterating over P, which is dim 1)
    for i in range(P):
        color = colors[i % len(colors)]
        
        # A. Anchor Points (Cluster)
        # Slicing: [All points, Class i, All spatial coords] -> (n_t, N)
        class_points = anchor_points[:, i, :]
        
        fig.add_trace(make_scatter(
            class_points, 
            mode='markers', 
            name=f'AP{i+1}',
            marker=dict(color=color)
        ))
        
        # B. Centers
        # Assumes self.centers is shape (P, N)
        fig.add_trace(make_scatter(
            centers[i], 
            mode='markers', 
            name=f'AP{i+1} Center',
            marker=dict(size=14, color=color, symbol='x')
        ))
        
        # C. Axes
        fig.add_trace(make_scatter(
            ap_axes[:,i,:], 
            mode='markers', 
            name=f'AP{i+1} Axis',
            marker=dict(size=14, color=color, opacity=0.5, symbol='circle-open')
        ))

    # 5. Layout Configuration
    layout_args = dict(
        height=400, width=600,
        title=f"Manifold Anchor Structure ({N}D)",
        template="plotly_white",
        margin=dict(l=40, r=40, b=40, t=60),
    )

    if N == 3:
        layout_args['scene'] = dict(  # type: ignore
            aspectmode='cube', # Essential for 3D geometry
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z'),
        )
    else:
        layout_args.update(dict(
            xaxis=dict(scaleanchor="y", scaleratio=1), # Lock aspect ratio for 2D
            yaxis=dict(constrain="domain")
        ))  # type: ignore

    fig.update_layout(**layout_args)  # type: ignore
    return fig


def plot_cone_geometry(P: int,
                       G_jnp: jnp.ndarray, 
                       M: int, 
                       t_k_jnp: jnp.ndarray, 
                       primal_jnp: jnp.ndarray,
                       anchor_points = None,
                       title=None):
   
    """
    Plots a figure showing data cones, projections, and anchor points.
    Automatically detects if data is 2D or 3D and renders accordingly.

    Args:
        P (int): Number of classes.
        G (jax.Array): Matrix of shape (P*M, N). N determines dimension (2 or 3).
        M (int): Number of points per class.
        t_ks (array-like): Random Probe vector.
        primal_jnp (array-like): Primal vector.
        anchor_points (jax.Array, optional): Matrix of anchor points.
        title: string for plot title
    """
    # --- 1. Data Preparation ---
    G = np.array(G_jnp)
    t_k = np.array(t_k_jnp).flatten()
    primal = np.array(primal_jnp).flatten()
    # print(f"{primal=}")
    w = t_k - primal # Moreau Decomposition: t = w + primal
    # print(f"{w=}")
    # print(f"{primal@w.T=}")
    # Detect Dimension (2 or 3)
    dim = G.shape[-1]
    # Reshape G to (P, M, dim)
    points_per_class = G.reshape(P, M, dim)

    # --- 2. Dynamic Range Calculation ---
    # Stack all coordinates to find global max for axis scaling
    all_points_list = [G, t_k.reshape(1, -1), primal.reshape(1, -1), w.reshape(1, -1)]
    if anchor_points is not None:
        anchor_np = np.array(anchor_points)
        all_points_list.append(anchor_np)

    all_coords = np.vstack(all_points_list)
    max_abs_val = np.max(np.abs(all_coords))
    range_val = max_abs_val + 0.5
    line_multiplier = range_val * 3.0 # Ensure lines go off-screen

    # Initialize Figure
    fig = go.Figure()

    # Color Palette
    base_colors = px.colors.qualitative.Plotly
    class_colors = (base_colors * (P // len(base_colors) + 1))[:P]

    # --- 3. Factories (The Smart Wrappers) ---
    
    def make_scatter(coords, **kwargs):
        """
        Factory: Returns go.Scatter or go.Scatter3d based on 'dim'.
        Automatically slices coords into x, y, (z).
        """
        # Ensure coords is 2D array even for single points
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
            
        common_args = dict(x=coords[:, 0], y=coords[:, 1], **kwargs)
        
        if dim == 3:
            return go.Scatter3d(z=coords[:, 2], **common_args)
        else:
            return go.Scatter(**common_args)

    def create_ray_lines(points, color, name, group):
        """
        Helper to generate the [Origin -> Point -> None] data structure 
        for drawing multiple lines in a single trace.
        """
        # Pre-allocate arrays including the `None` spacers
        # Shape: (NumPoints * 3, Dim) -> [0,0], [x,y], [None, None]...
        num_points = points.shape[0]
        ray_coords = np.full((num_points * 3, dim), np.nan) 
        
        # Fill Origins (0,0,...)
        ray_coords[0::3] = 0 
        # Fill Endpoints (point * multiplier)
        ray_coords[1::3] = points * line_multiplier
        # The 3rd indices [2::3] remain NaN to break the lines
        
        return make_scatter(
            ray_coords,
            mode='lines',
            line=dict(color=color, width=1, dash='dash'),
            opacity=0.3,
            name=f"{name} Cone",
            legendgroup=group,
            hoverinfo='skip'
        )

    # --- 4. Add Traces ---

    # A. Classes (Points + Cones)
    for i in range(P):
        current_points = points_per_class[i]
        color = class_colors[i]
        group_id = f"group{i}"
        
        # 1. Cone Rays
        fig.add_trace(create_ray_lines(current_points, color, f"Class {i+1}", group_id))
        
        # 2. Points
        fig.add_trace(make_scatter(
            current_points,
            mode='markers',
            name=f"Class {i+1} Points",
            legendgroup=group_id,
            marker=dict(size=5 if dim==3 else 10, color=color) # Smaller markers for 3D
        ))

    # B. Vectors
    # Helper to add single vectors easily
    def add_vec(vec, color, name):
        # Create a line from origin to vector
        origin = np.zeros((1, dim))
        line_data = np.vstack([origin, vec.reshape(1, dim)])
        
        fig.add_trace(make_scatter(
            line_data,
            mode='lines+markers',
            line=dict(color=color, width=4),
            marker=dict(size=10, color=color, symbol='circle'),
            name=name
        ))

    add_vec(t_k, 'purple', 'Random Probe (t)')
    add_vec(primal, 'green', 'Primal (x)')
    add_vec(w, 'orange', 'Dual/Polar (w)')

    # C. Anchor Points
    if anchor_points is not None:
        fig.add_trace(make_scatter(
            anchor_np,
            mode='markers',
            name="Anchor Points",
            marker=dict(size=12, color='black', opacity=0.5, symbol='cross')
        ))

    # --- 5. Layout Factory ---
    
    layout_args = dict(
        height=400, width=600,
        title=title,
        template="plotly_white",
        margin=dict(l=20, r=20, b=20, t=60),
    )

    if dim == 3:
        # 3D Specific Layout
        scene_dict = dict(
            xaxis=dict(range=[-range_val, range_val]),
            yaxis=dict(range=[-range_val, range_val]),
            zaxis=dict(range=[-range_val, range_val]),
            aspectmode='cube' # Crucial for 3D geometry to look correct
        )
        layout_args['scene'] = scene_dict
    else:
        # 2D Specific Layout
        layout_args.update(dict(
            xaxis=dict(range=[-range_val, range_val], zeroline=True, zerolinewidth=1, zerolinecolor='black'),
            yaxis=dict(
                range=[-range_val, range_val], 
                scaleanchor="x", 
                scaleratio=1,
                zeroline=True, zerolinewidth=1, zerolinecolor='black'
            ),
        ))

    fig.update_layout(**layout_args) # type: ignore
    return fig


def simulated_geometry():
    """Brute force method for finding capacity, should align with glue/simulated capacity"""
    pass



def run_glue_analysis_pipeline(glue_metrics, config):
    """
    Analyzes, saves, and plots the GLUE metrics returned by run_glue_solver.
    
    Args:
        glue_metrics (dict): Dictionary of metric histories. 
                             Values should be numpy arrays of shape (Steps, Repeats) or (Steps,).
                             Example: {'test_accuracy': array(...), 'forgetting': array(...)}
        config: Configuration object containing 'dataset_name', 'figures_dir', 
                'log_frequency', 'num_tasks', 'epochs_per_task', etc.
    """
    print("--- Starting GLUE Metric Analysis ---")
    
    # 1. Setup Directories
    # We save the raw metric data in the same reps_dir or a results folder
    save_dir = getattr(config, 'reps_dir', os.path.join("results", config.dataset_name, config.algorithm))
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(config.figures_dir, exist_ok=True)
    
    # 2. Archive Raw Metrics
    save_data_path = os.path.join(save_dir, f"glue_metrics_{config.dataset_name}.pkl")
    
    # Calculate task boundaries for saving/plotting
    # Assuming standard continual learning setup: Total Epochs = Tasks * Epochs_per_Task
    task_boundaries = []
    if hasattr(config, 'num_tasks') and hasattr(config, 'epochs_per_task'):
        cumulative_epochs = 0
        for _ in range(config.num_tasks):
            cumulative_epochs += config.epochs_per_task
            task_boundaries.append(cumulative_epochs)
    
    with open(save_data_path, 'wb') as f:
        pickle.dump({'metrics': glue_metrics, 'task_boundaries': task_boundaries}, f)
    print(f"GLUE metrics data saved to {save_data_path}")

    # 3. Filter Plottable Metrics
    # We only plot items that are time-series (lists or numpy arrays)
    plot_metrics = {}
    for k, v in glue_metrics.items():
        if isinstance(v, (list, np.ndarray)):
            arr = np.array(v)
            # Ensure it has at least 1 dimension
            if arr.ndim > 0:
                plot_metrics[k] = arr
    
    if not plot_metrics:
        print("No plottable time-series metrics found in output.")
        return

    metric_names = list(plot_metrics.keys())
    n_metrics = len(metric_names)
    
    # 4. Generate Plots
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
    if n_metrics == 1: 
        axes = [axes]
    
    # Determine X-axis (Epochs)
    # Use the length of the first metric to determine total steps
    first_metric_data = plot_metrics[metric_names[0]]
    total_steps = first_metric_data.shape[0]
    
    # Construct x-axis based on log frequency
    x_axis = np.arange(1, total_steps + 1) * config.log_frequency
    
    print(f"Plotting {n_metrics} metrics over {total_steps} logging steps...")

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        data = plot_metrics[metric]
        
        # Calculate Mean and Std Dev if multiple repeats exist
        # Expected shape: (Steps, Repeats)
        if data.ndim > 1 and data.shape[1] > 1:
            mean = np.nanmean(data, axis=1)
            std = np.nanstd(data, axis=1)
            
            ax.plot(x_axis, mean, label='Mean', color='purple', linewidth=2)
            ax.fill_between(x_axis, mean - std, mean + std, color='purple', alpha=0.2)
        else:
            # Single run or flattened
            flat_data = data.flatten() if data.ndim > 1 else data
            ax.plot(x_axis, flat_data, color='purple', linewidth=2)

        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Add Vertical Lines for Task Boundaries
        # We assume boundaries are in 'Epochs', so they align directly with x_axis
        for boundary in task_boundaries[:-1]: # Skip the very last boundary (end of training)
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)

        if i == 0:
            ax.set_title(f"GLUE Metrics ({config.dataset_name}) - {config.algorithm}")
            if data.ndim > 1 and data.shape[1] > 1:
                ax.legend(loc='upper right')

    axes[-1].set_xlabel('Epochs')
    plt.tight_layout()
    
    # 5. Save Plot
    save_plot_path = os.path.join(config.figures_dir, f"glue_metrics_{config.dataset_name}.png")
    plt.savefig(save_plot_path, dpi=150)
    plt.close()
    print(f"GLUE metric plots saved to {save_plot_path}")


