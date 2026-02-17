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

@jax.jit(static_argnames=('P', 'M', 'N', 'n_t', 'qp_solver'))
def run_glue_solver(key, data: jax.Array, P:int, M:int, N:int, n_t:int, qp_solver):
    m_key, t_key, y_key = jax.random.split(key, 3)
    A = jnp.eye(N)
    H = jnp.zeros((P*M,1))
    m_data = get_m_data(m_key, data, P, M, N)
    all_ts = get_all_ts(t_key, N, n_t)
    all_ys = get_all_ys(y_key, P, n_t)
    P_indices = jnp.array(list(permutations(range(P), r=2)))
    # solve the qp equation
    anchor_points, Gs, Primals = sample_anchor_points(m_data, all_ts, all_ys, A, H, qp_solver) # (aps, Gs, primals), shapes: ((n_t, P,N),(n_t,P*M,N),(n_t,N,1))
    centers, center_gram = get_ap_centers(anchor_points)
    ap_axes, t_1ks, axes_gram = get_aps_axis_var(anchor_points, centers, all_ts)

    # manifold geometries
    capacity = (1/P * jnp.mean(jax.vmap(lambda s,t: (s@t.T).T @ jnp.linalg.pinv(s@s.T) @ (s@t.T), in_axes=(0,0))(anchor_points, all_ts)))**(-1)
    dimension = (1/P * jnp.mean(jax.vmap(lambda t,g: t.T @ jnp.linalg.pinv(g) @ t, in_axes=(0,0))(t_1ks, axes_gram)))
    indiv_dim = jnp.mean(jax.vmap(lambda t,g: t * (jnp.linalg.pinv(g)@t), in_axes=(0,0))(t_1ks, axes_gram), axis=0)
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


def get_m_data(key, data: jax.Array, P: int, M, N) -> jax.Array:
    """Sample from full data to get M data points
    data shape: (P classes, num Points, N features)"""
    m_keys = jax.random.split(key, P)
    n_data_points = data.shape[1]
    def sample_m(k):
        return jax.random.choice(k, n_data_points, shape=(M,), replace=False)
    m_data_axes = jax.vmap(sample_m)(m_keys)
    m_data = jax.vmap(lambda d, indcs: d[indcs,:])(data, m_data_axes)
    assert m_data.shape[0] == P, "M_data wrong dim 0 size"
    assert m_data.shape[1] == M, "M_data wrong dim 1 size"
    assert m_data.shape[2] == N, "M_data wrong dim 2 size"
    return m_data


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
        return t.T @ jnp.linalg.pinv(g_1+G_0) @ t
    def rad_bot(t,g_1):
        return t.T @ jnp.linalg.pinv(g_1 + (g_1 @ jnp.linalg.pinv(G_0) @ g_1)) @ t
    # for neural data chou2025a
    top_values = jax.vmap(rad_top)(t_1ks, G_1ks)
    bot_values = jax.vmap(rad_bot)(t_1ks, G_1ks)
    mean_top = jnp.mean(top_values)
    mean_bot = jnp.mean(bot_values)
    radius = jnp.sqrt(mean_top/mean_bot)
    # for neural networks representations
    # radius = jnp.sqrt(jnp.mean(jax.vmap(lambda t,g: rad_top(t,g)/rad_bot(t,g), in_axes=(0,0))(t_1ks, G_1ks)))
    return radius

def get_indiv_radius(t_1ks, G_1ks, G_0):
    def rad_top(t,g_1):
        return t.T * jnp.linalg.pinv(g_1+G_0) @ t
    def rad_bot(t,g_1):
        return t.T * jnp.linalg.pinv(g_1 + (g_1 @ jnp.linalg.pinv(G_0) @ g_1)) @ t
    # for neural data chou2025a
    top_values = jax.vmap(rad_top)(t_1ks, G_1ks)
    bot_values = jax.vmap(rad_bot)(t_1ks, G_1ks)
    mean_top = jnp.mean(top_values, axis=0)
    mean_bot = jnp.mean(bot_values, axis=0)
    radius = jnp.sqrt(mean_top/mean_bot)
    # for neural networks representations
    # radius = jnp.sqrt(jnp.mean(jax.vmap(lambda t,g: rad_top(t,g)/rad_bot(t,g), in_axes=(0,0))(t_1ks, G_1ks)))
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


##### old class version

# class glue_solver():
#     def __init__(self, glue_key, 
#                  P_point_clouds: int, 
#                  m_points: int, 
#                  n_dim_space: int, 
#                  n_t: int,
#                  one_v_rest: bool = True):
#         """Take the hyperparameters and setup the t and y values"""
#         self.P = P_point_clouds
#         self.M = m_points
#         self.N = n_dim_space
#         self.n_t = n_t
#         self.A = jnp.eye(self.N)
#         self.H = jnp.zeros((self.P*self.M,1))
#         t_key, y_key, self.main_key = jax.random.split(glue_key, 3)
#         # set up probes and dichotomies
#         t_mu = jnp.zeros(self.N)
#         t_sigma = jnp.eye(self.N)

#         # use glue key to generate t and y keys
#         self.all_t_ks = jax.random.multivariate_normal(t_key, t_mu, t_sigma, shape=(self.n_t,))
#         self.all_y_ks = self.get_dichotomies(y_key, one_v_rest)
#         self.P_indices = jnp.array(list(permutations(range(self.P), r=2)))

#     def get_one_v_rest_ys(self, ys_key) -> jax.Array:
#         # 1. Initialize an array of -1s
#         potential_ys = jnp.full((self.n_t, self.P), -1)
#         # 2. Generate random indices for the single '1' in each row
#         # Each index will be between 0 and P-1
#         one_indices = jax.random.randint(ys_key, shape=(self.n_t,), minval=0, maxval=self.P)
#         # 3. Use scatter update to place the 1s at those indices
#         row_indices = jnp.arange(self.n_t)
#         potential_ys = potential_ys.at[row_indices, one_indices].set(1)
#         return potential_ys
    
#     def entropy_dichotomy(self, ys_key) -> jax.Array:
#         ychoice_k, yperm_k = jax.random.split(ys_key)
#         labels = jnp.array([1,-1])
#         potential_ys = jax.random.choice(ychoice_k, 
#                                         labels, 
#                                         shape=(self.n_t, self.P), 
#                                         replace=True, 
#                                         mode="low")
#         ys_filler = jnp.concatenate([labels]*self.P)[:self.P]
#         non_dichotomy_mask = jnp.abs(jnp.sum(potential_ys, axis=1))==self.P
#         n_fillers = int(jnp.sum(non_dichotomy_mask))
#         if n_fillers > 0:
#             repeated_filler = jnp.stack([ys_filler,]*n_fillers, axis=0)
#             shuffled_filler = jax.random.permutation(yperm_k, repeated_filler, axis=1, independent=True)
#             potential_ys = potential_ys.at[non_dichotomy_mask,:].set(shuffled_filler)
#         return potential_ys

#     def get_dichotomies(self, ys_key, one_v_rest: bool)->jax.Array:
#         """Return the dichotomies for P classes for each n_t
#             returns a matrix: n_t x P filled with 1/-1 """
#         potential_ys = self.get_one_v_rest_ys(ys_key) if one_v_rest else self.entropy_dichotomy(ys_key)
#         assert potential_ys.shape[0] == self.n_t, "n_t shape for potential ys incorrect"
#         assert potential_ys.shape[1] == self.P, "P shape for potnential ys incorrect"
#         return potential_ys 
    
#     def get_m_data(self, data) -> jax.Array:
#         """Sample from full data to get M data points
#         data shape: (P classes, num Points, N features)"""
#         if type(data) != jax.Array:
#             data = jnp.array(data)
#         m_key, self.main_key = jax.random.split(self.main_key)
#         m_keys = jax.random.split(m_key, self.P)
#         n_data_points = data.shape[1]
#         def sample_m(k):
#             return jax.random.choice(k, n_data_points, shape=(self.M,), replace=False)
#         m_data_axes = jax.vmap(sample_m)(m_keys)
#         m_data = jax.vmap(lambda d, indcs: d[indcs,:])(data, m_data_axes)
#         assert m_data.shape[0] == self.P, "M_data wrong dim 0 size"
#         assert m_data.shape[1] == self.M, "M_data wrong dim 1 size"
#         assert m_data.shape[2] == self.N, "M_data wrong dim 2 size"
#         return m_data

#     def run(self, data):
#         """runs glue solver: take plots, make G, 
#         calc anchor point 
#         -> ap centers, axis -> capacity, dim, radius, etc.
#         Plot if true, if dimensions are larger than 3, 
#         embed them using PCA, then plot
#         inputs: jax array, shape: (P,D,N), where D is the number of datapoints in the data
#         """
#         assert data.shape[0] == self.P, "Data has incorrect P classes"
#         assert data.shape[1] >= self.M, "Data has insufficient M data points"
#         assert data.shape[2] == self.N, "Data has incorrect N features"

#         m_data = self.get_m_data(data)
#         anchor_points, Gs, Primals = sample_anchor_points(m_data, self.all_t_ks, self.all_y_ks, self.A, self.H) # (aps, Gs, primals), shapes: ((n_t, P,N),(n_t,P*M,N),(n_t,N,1))
#         centers, center_gram = get_ap_centers(anchor_points)
#         s_1ks, t_1ks, G_1k = get_aps_axis_var(anchor_points, centers, self.all_t_ks)
#         self.anchor_points = anchor_points
#         self.Gs = Gs
#         self.Primals = Primals
#         self.centers = centers
#         self.center_gram = center_gram
#         self.ap_axes = s_1ks
#         self.probe_axes_projs = t_1ks
#         self.axes_gram = G_1k
#         geometries = self.manifold_geometries()
#         self.geometries = geometries
#         return geometries

#     def get_radius(self, t_1ks, G_1ks, G_0):
#         def rad_top(t,g_1):
#             return t.T @ jnp.linalg.pinv(g_1+G_0) @ t
#         def rad_bot(t,g_1):
#             return t.T @ jnp.linalg.pinv(g_1 + (g_1 @ jnp.linalg.pinv(G_0) @ g_1)) @ t
#         radius = jnp.sqrt(jnp.mean(jax.vmap(lambda t,g: rad_top(t,g)/rad_bot(t,g), in_axes=(0,0))(t_1ks, G_1ks)))
#         return radius
    
#     def get_center_alignment(self, axis_centers):
#         inner_f = lambda ac,pi: (ac[pi[0]].T @ ac[pi[1]])/(jnp.linalg.norm(ac[pi[0]]) * jnp.linalg.norm(ac[pi[1]]))
#         center_alignment = 1/(self.P * (self.P -1)) * jnp.sum(
#             jax.vmap(inner_f, 
#                         in_axes=(None,0))(axis_centers, self.P_indices))
#         return center_alignment
    
#     def get_axis_alignment(self, all_aps_axis):
#         # all_aps_axis shape: (n_t,P,N)
#         axis_alignment = 1/(self.P*(self.P-1)) * jnp.sum(
#             jax.vmap(
#                 lambda x, pid: axis_align_nt_sum(x[:,pid[0],:],x[:,pid[1],:]),
#                 in_axes=(None, 0))
#                 (all_aps_axis, self.P_indices)
#                 )
#         return axis_alignment

#     def get_center_axis_alignment(self, aps_centers, aps_axes):
#         center_axis_alignment = 1/(self.P*(self.P-1)) * jnp.sum(
#             jax.vmap(
#                 lambda x,y, pid: center_axis_align_nt_sum(x[pid[0]],y[:, pid[1], :]),
#                 in_axes=(None, None, 0))
#                 (aps_centers, aps_axes, self.P_indices)
#                 )
#         return center_axis_alignment
    
#     def manifold_geometries(self):
#         capacity = (1/self.P * jnp.mean(jax.vmap(lambda s,t: (s@t.T).T @ jnp.linalg.pinv(s@s.T) @ (s@t.T), in_axes=(0,0))(self.anchor_points, self.all_t_ks)))**(-1)
#         dimension = (1/self.P * jnp.mean(jax.vmap(lambda t,g: t.T @ jnp.linalg.pinv(g) @ t, in_axes=(0,0))(self.probe_axes_projs, self.axes_gram)))
#         radius = self.get_radius(self.probe_axes_projs, self.axes_gram, self.center_gram)
#         center_align = self.get_center_alignment(self.centers)
#         axis_align = self.get_axis_alignment(self.ap_axes)
#         center_axis_align = self.get_center_axis_alignment(self.centers, self.ap_axes)
#         approx_capacity = (1 + radius**(-2))/dimension
#         return capacity, dimension, radius, center_align, axis_align, center_axis_align, approx_capacity

#     def simulated_capacity(self):
#         """Brute force method for finding capacity, should align with glue/simulated capacity"""

#         return

#     def individual_geometries(self, *args, **kwargs):
#         """use chou2025a line 1552 equations for calculating individual class dimensions, indexing into P
#         Make sure to always align with the P so as to track correct classes"""
#         pass

#     def make_plots(self, plots_path: str| None = None, n_plots: int=3):
#         # Make 3 cone plots, 3 columns 1 row
#         # Make 1 anchor point plot, APs, centers and axis maybe with points
#         # want logic to handle N= 2 or 3 and TODO: if more than do PCA to embed. 
#         # Ensure directory exists
#         if plots_path:
#             base_dir = os.path.dirname(plots_path)
#             if base_dir and not os.path.exists(base_dir):
#                 os.makedirs(base_dir)

#             html_filename = f"{plots_path}/manifold_plots.html"

#         k = min(n_plots, self.n_t)
#         n_t_sample_indcs = sample(range(self.n_t), k)
#         js_str = 'cdn'
#         figs = []
#         for i in n_t_sample_indcs:
#             title_str = f"Geometries for n_t: {i}; y_k: {np.array(self.all_y_ks[i])}"
#             figs.append(plot_cone_geometry(self.P, 
#                                     self.Gs[i], 
#                                     self.M, 
#                                     self.all_t_ks[i],
#                                     self.Primals[i],
#                                     self.anchor_points[i],
#                                     title=title_str))
            
#         figs.append(self.make_manifold_plot())
#         if plots_path:
#             for fig in figs:
#                 # html plot
#                 with open(html_filename, 'w') as f:
#                     f.write(fig.to_html(full_html=False, include_plotlyjs=js_str))
#                     js_str = False
#                 # png plot
#                 png_name = f"{plots_path}/manifold_plot_n_t{i}.png"
#                 fig.write_image(png_name)
#         return figs
    
#     def make_manifold_plot(self):
#         """
#         Plots anchor points, centers, and axes. 
#         self.anchor_points shape: (n_t, P, N)
#         self.ap_axes shape: (n_t, P, N)
#         self.centers shape: (P, N)
#         - n_t: number of anchor points per class
#         - P: number of classes
#         - N: dimension (2 or 3)
#         """

#         # 1. Detect Dimension and Classes
#         # Shape is (n_t, P, N)
#         n_t, num_classes, dim = self.anchor_points.shape

#         # 2. Trace Factory (Smart Wrapper)
#         def make_scatter(coords, **kwargs):
#             # Ensure coords is at least 2D array (1, dim) or (n_t, dim)
#             if coords.ndim == 1:
#                 coords = coords.reshape(1, -1)
                
#             common_args = dict(x=coords[:, 0], y=coords[:, 1], **kwargs)
            
#             if dim == 3:
#                 return go.Scatter3d(z=coords[:, 2], **common_args)
#             else:
#                 return go.Scatter(**common_args)

#         # 3. Initialize Figure
#         fig = go.Figure()
        
#         # Color palette to match P classes
#         colors = px.colors.qualitative.Plotly
        
#         # 4. Loop over classes (Iterating over P, which is dim 1)
#         for i in range(num_classes):
#             color = colors[i % len(colors)]
            
#             # A. Anchor Points (Cluster)
#             # Slicing: [All points, Class i, All spatial coords] -> (n_t, N)
#             class_points = self.anchor_points[:, i, :]
            
#             fig.add_trace(make_scatter(
#                 class_points, 
#                 mode='markers', 
#                 name=f'AP{i+1}',
#                 marker=dict(color=color)
#             ))
            
#             # B. Centers
#             # Assumes self.centers is shape (P, N)
#             fig.add_trace(make_scatter(
#                 self.centers[i], 
#                 mode='markers', 
#                 name=f'AP{i+1} Center',
#                 marker=dict(size=14, color=color, symbol='x')
#             ))
            
#             # C. Axes
#             fig.add_trace(make_scatter(
#                 self.ap_axes[:,i,:], 
#                 mode='markers', 
#                 name=f'AP{i+1} Axis',
#                 marker=dict(size=14, color=color, opacity=0.5, symbol='circle-open')
#             ))

#         # 5. Layout Configuration
#         layout_args = dict(
#             height=400, width=600,
#             title=f"Manifold Anchor Structure ({dim}D)",
#             template="plotly_white",
#             margin=dict(l=40, r=40, b=40, t=60),
#         )

#         if dim == 3:
#             layout_args['scene'] = dict(  # type: ignore
#                 aspectmode='cube', # Essential for 3D geometry
#                 xaxis=dict(title='x'),
#                 yaxis=dict(title='y'),
#                 zaxis=dict(title='z'),
#             )
#         else:
#             layout_args.update(dict(
#                 xaxis=dict(scaleanchor="y", scaleratio=1), # Lock aspect ratio for 2D
#                 yaxis=dict(constrain="domain")
#             ))  # type: ignore

#         fig.update_layout(**layout_args)  # type: ignore
#         return fig



############# OlD version ######################
# # --- JAX-JIT Compiled Optimization Kernels ---

# @jax.jit(static_argnums=(2, 3, 4))
# def solve_single_anchor(key, flat_manifolds, M, P, N):
#     """
#     Solves for a single anchor point (s_i) and direction (t).
#     Pure JAX implementation of the dual SVM problem.
#     """
#     # 1. Sample direction t and dichotomy y
#     key, t_key, y_key = jax.random.split(key, 3)
    
#     # t_k ~ N(0, I_N)
#     t = jax.random.normal(t_key, (N,)) 
    
#     # Random dichotomy y ~ {-1, 1}
#     y = jax.random.choice(y_key, jnp.array([-1.0, 1.0]), (P,))
    
#     # 2. Formulate QP
#     # Obj: min 1/2 ||x - t||^2  => min 1/2 x'Ix - t'x
#     Q_mat = jnp.eye(N)
#     q_vec = -t
    
#     # Constraints: y_i * (point_j . x) >= 0 (Margin condition)
#     # Standard QP form in OSQP is: l <= Ax <= u (This wrapper uses Gx <= h)
#     # So: -(y_i * point_j) . x <= 0
#     y_expanded = jnp.repeat(y, M) # (P*M,)
    
#     # Negate G_mat to enforce positive margin
#     G_mat = -(y_expanded[:, None] * flat_manifolds) # (P*M, N)
#     h_vec = jnp.zeros((P * M,))
    
#     # OSQP Solver
#     solver = OSQP(tol=1e-5, maxiter=4000, verbose=False)
#     sol = solver.run(params_obj=(Q_mat, q_vec), params_eq=None, params_ineq=(G_mat, h_vec))
    
#     # 3. Recover Anchor Points s_i from Dual Variables
#     z = sol.params.dual_ineq
#     z = jnp.maximum(z, 0.0)
    
#     z_reshaped = z.reshape(P, M)
#     z_sums = jnp.sum(z_reshaped, axis=1, keepdims=True)
    
#     # Normalize duals to get weights (alphas) for each manifold
#     safe_sums = z_sums + 1e-10
#     alphas = z_reshaped / safe_sums
    
#     manifolds = flat_manifolds.reshape(P, M, N)
    
#     # s_i_raw: The actual point on the manifold (weighted average of support vectors)
#     s_i_raw = jnp.einsum('pm,pmn->pn', alphas, manifolds)
    
#     return s_i_raw, t


# @jax.jit
# def compute_metrics_from_anchors(anchors_raw, t_vectors):
#     """
#     Computes GLUE metrics from the solved anchor points.
#     """
#     n_t, P, N = anchors_raw.shape
    
#     # --- 1. Decomposition into Centers and Axes ---
#     s_0 = jnp.mean(anchors_raw, axis=0) # (P, N) - Manifold Centroids
    
#     # Axis s^1: Deviation from center
#     s_1 = anchors_raw - s_0[None, :, :]
    
#     # --- 2. Gram Matrix Projections per sample k ---
#     def process_single_sample(s1_k, t_k):
#         v_axis = s1_k @ t_k 
#         G_axis = s1_k @ s1_k.T
#         G_axis_inv = jnp.linalg.pinv(G_axis, rcond=1e-5)
#         norm_proj_axis_sq = v_axis.T @ G_axis_inv @ v_axis
#         return norm_proj_axis_sq

#     norm_proj_axis_sq = jax.vmap(process_single_sample)(s_1, t_vectors)
    
#     # --- 3. Aggregate Metrics ---
#     D_M = jnp.mean(norm_proj_axis_sq)
    
#     norm_s1 = jnp.mean(jnp.sum(s_1**2, axis=-1)) # Mean squared norm of axes
#     norm_s0 = jnp.mean(jnp.sum(s_0**2, axis=-1)) # Mean squared norm of centers
    
#     R_M_val = jnp.sqrt(norm_s1 / (norm_s0 + 1e-9))

#     # Capacity Approximation Formula (Chung et al.)
#     def compute_capacity(r, d):
#         return (1.0 + 1.0 / (r**2)) / d
    
#     def return_nan(r, d):
#         return jnp.nan
    
#     # Check if metrics are valid (not too small, not NaN)
#     is_valid = (R_M_val > 1e-6) & (D_M > 1e-6) & jnp.isfinite(R_M_val) & jnp.isfinite(D_M)
#     capacity = jax.lax.cond(is_valid, compute_capacity, return_nan, R_M_val, D_M)

#     # --- 4. Alignment Metrics ---
#     norms_0 = jnp.linalg.norm(s_0, axis=-1, keepdims=True)
#     s_0_normed = s_0 / (norms_0 + 1e-9)
#     cos_sim_0 = jnp.abs(s_0_normed @ s_0_normed.T)
#     mask = 1.0 - jnp.eye(P)
#     rho_c = jnp.sum(cos_sim_0 * mask) / (P * (P - 1))
    
#     def compute_axis_corr(s1_k):
#         norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
#         s1_normed = s1_k / (norms_1 + 1e-9)
#         cos_sim_1 = jnp.abs(s1_normed @ s1_normed.T)
#         return jnp.sum(cos_sim_1 * mask) / (P * (P - 1))
    
#     rho_a = jnp.mean(jax.vmap(compute_axis_corr)(s_1))
    
#     def compute_ca_corr(s1_k):
#         norms_1 = jnp.linalg.norm(s1_k, axis=-1, keepdims=True)
#         s1_normed = s1_k / (norms_1 + 1e-9)
#         cross_sim = jnp.abs(s_0_normed @ s1_normed.T)
#         return jnp.sum(cross_sim * mask) / (P * (P - 1))
        
#     psi = jnp.mean(jax.vmap(compute_ca_corr)(s_1))
    
#     return {
#         'Capacity': capacity,
#         'Radius': R_M_val,
#         'Dimension': D_M,
#         'Center_Alignment': rho_c,
#         'Axis_Alignment': rho_a,
#         'Center_Axis_Alignment': psi
#     }

# # --- Simulated Capacity Kernels (Algorithm 1) ---

# # indices: 0:key, 1:flat_manifolds, 2:n_proj, 3:M_per_manifold, 4:P
# def check_linear_separability_batch(key, flat_manifolds, n_proj, M_per_manifold, P):
#     """
#     Checks if randomly projected manifolds are linearly separable.
#     """
#     N_ambient = flat_manifolds.shape[1]
    
#     # 1. Random Projection: Pi from R^N -> R^n
#     k1, k2 = jax.random.split(key)
#     Pi = jax.random.normal(k1, (N_ambient, n_proj)) / jnp.sqrt(n_proj)
    
#     # Project data
#     projected_data = flat_manifolds @ Pi
    
#     # 2. Random Dichotomy
#     y = jax.random.choice(k2, jnp.array([-1.0, 1.0]), (P,))
#     y_expanded = jnp.repeat(y, M_per_manifold) # (P*M,)
    
#     # 3. Solve Linear Separability (SVM-like feasibility)
#     G_mat = -(y_expanded[:, None] * projected_data) # (P*M, n)
#     h_vec = -jnp.ones((P * M_per_manifold,))
    
#     Q_mat = jnp.eye(n_proj)
#     q_vec = jnp.zeros((n_proj,))
    
#     solver = OSQP(tol=1e-3, maxiter=1000, verbose=False, check_primal_dual_infeasability=True)
#     sol = solver.run(params_obj=(Q_mat, q_vec), params_eq=None, params_ineq=(G_mat, h_vec))
    
#     # Verify margin manually
#     w_opt = sol.params.primal
#     margins = y_expanded * (projected_data @ w_opt)
    
#     # If minimum margin is >= 1 - epsilon, it is separable
#     is_separable = jnp.min(margins) >= 0.99 
#     return is_separable

# def estimate_separability_probability(key, flat_manifolds, n_proj, M: int, P: int, m_trials=100):
#     """Estimates p_n: Probability that manifolds are separable in n dimensions."""
#     keys = jax.random.split(key, m_trials)
#     check_fn = lambda k: check_linear_separability_batch(k, flat_manifolds, n_proj, M, P)
#     results = jax.vmap(check_fn)(keys)
#     return jnp.mean(results.astype(jnp.float32))

# def compute_simulated_capacity(rng, representations, labels):
#     """
#     Implements the Binary Search algorithm (Algorithm 1) to find Simulated Capacity.
#     Fully uses JAX for random keys and array manipulation.
#     """
#     # Eager execution for data organization (dynamic shapes)
#     unique_labels = jnp.unique(labels)
#     P = unique_labels.shape[0]
    
#     if P < 2: return jnp.nan
    
#     # Organize data (ensure equal points M per manifold)
#     counts = jnp.array([jnp.sum(labels == l) for l in unique_labels])
#     M = int(jnp.min(counts))
    
#     if M < 2: return jnp.nan
    
#     grouped_data = []
#     # Loop over classes in standard python, but operations are JAX
#     for i in range(P):
#         l = unique_labels[i]
#         # Boolean masking returns dynamic shape, handled by JAX eager execution
#         idxs = jnp.where(labels == l)[0][:M]
#         grouped_data.append(representations[idxs])
    
#     manifolds = jnp.stack(grouped_data) # (P, M, N)
#     flat_manifolds = manifolds.reshape(-1, manifolds.shape[-1])
#     N_ambient = flat_manifolds.shape[1]

#     # Binary Search for n*
#     # We want smallest n such that p_n >= 0.5
#     n_left = 1
#     n_right = N_ambient
#     n_star = N_ambient
    
#     iteration = 0
#     while n_left <= n_right:
#         n_mid = (n_left + n_right) // 2
#         if n_mid == 0: n_mid = 1
        
#         # Split key for this iteration's trials
#         rng, iter_key = jax.random.split(rng)
        
#         p_n = estimate_separability_probability(
#             iter_key, 
#             flat_manifolds, 
#             int(n_mid), 
#             int(M), 
#             int(P), 
#             m_trials=100
#         )
        
#         if p_n >= 0.5:
#             n_star = n_mid
#             n_right = n_mid - 1 
#         else:
#             n_left = n_mid + 1 
            
#         iteration += 1
        
#     alpha_sim = P / n_star
#     return float(alpha_sim)


# def run_manifold_geometry(rng, representations, labels, n_samples_t=50):
#     """
#     Main entry point for computing manifold metrics (GLUE).
#     """
#     # Check for NaNs
#     if not jnp.all(jnp.isfinite(representations)):
#         print("Warning: NaN or Inf detected in representations")
#         return {k: jnp.nan for k in ['Capacity', 'Radius', 'Dimension', 
#                                    'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']}
    
#     unique_labels = jnp.unique(labels)
#     P = unique_labels.shape[0]
#     N = representations.shape[1]
    
#     counts = jnp.array([jnp.sum(labels == l) for l in unique_labels])
#     M = int(jnp.min(counts))
    
#     if M < 2:
#         return {k: jnp.nan for k in ['Capacity', 'Radius', 'Dimension', 
#                                    'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']}

#     grouped_data = []
#     for i in range(P):
#         l = unique_labels[i]
#         idxs = jnp.where(labels == l)[0][:M]
#         grouped_data.append(representations[idxs])
    
#     manifolds = jnp.stack(grouped_data)
#     flat_manifolds = manifolds.reshape(-1, N)
    
#     # Generate keys for the solver
#     keys = jax.random.split(rng, n_samples_t)
#     solve_batch = jax.vmap(solve_single_anchor, in_axes=(0, None, None, None, None))
    
#     # Run Solver
#     s_raw, t_vecs = solve_batch(keys, flat_manifolds, M, P, N)
#     metrics_jax = compute_metrics_from_anchors(s_raw, t_vecs)
    
#     return {k: float(v) for k, v in metrics_jax.items()}

# # --- Analysis Pipeline Integration ---

# def analyze_manifold_trajectory(config, task_names):
#     """
#     Loads saved representations and computes manifold metrics over time.
#     Uses JAX for all computations and key management.
#     """
#     print("\n--- Starting Manifold Geometric Analysis (GLUE) [JAX Backend] ---")
    
#     # Initialize the master key from config seed
#     master_key = jax.random.PRNGKey(config.seed)

#     glue_metrics = ['Capacity', 'Radius', 'Dimension', 
#                     'Center_Alignment', 'Axis_Alignment', 'Center_Axis_Alignment']
    
#     extra_metrics = ['Simulated_Capacity', 'Capacity_Relative_Error']
#     all_stored_metrics = glue_metrics + extra_metrics

#     # Dictionary to store full results per task
#     full_results = {}
    
#     # For plotting aggregate history (only GLUE metrics)
#     plot_history = {k: [] for k in glue_metrics}
    
#     # Store Simulated Capacity Results for Overlay
#     sim_cap_history = {}
    
#     task_boundaries = []
#     current_epoch = 0
    
#     for t_idx, t_name in enumerate(task_names):
#         rep_path = os.path.join(config.reps_dir, f"{t_name}_reps_per_epoch.npy")
#         lbl_path = os.path.join(config.reps_dir, f"{t_name}_labels.npy")
        
#         if not os.path.exists(rep_path) or not os.path.exists(lbl_path):
#             print(f"Skipping {t_name} (files not found)")
#             continue
            
#         # Load using JAX (via numpy then convert)
#         with open(rep_path, 'rb') as f:
#             reps_data = jnp.load(f)
#         with open(lbl_path, 'rb') as f:
#             labels = jnp.load(f)    
        
#         # Expecting reps_data: (Steps, Repeats, Samples, Dim)
#         n_steps, n_repeats, n_samples, dim = reps_data.shape
        
#         # Asserts for system integrity
#         assert n_repeats == config.n_repeats, f"Config repeats {config.n_repeats} mismatch data {n_repeats}"
        
#         # Handle Labels: Expect (Samples, Repeats) from single_run
#         if labels.ndim == 2:
#             assert labels.shape == (n_samples, n_repeats), f"Label shape {labels.shape} mismatch (N, P)"
#             labels_reshaped = labels
#         else:
#             # Fallback if flattened
#             assert labels.size == n_samples * n_repeats
#             labels_reshaped = labels.reshape(n_samples, n_repeats)
        
#         print(f"Processing {t_name}: {n_steps} steps, {n_repeats} repeats...")
        
#         # Storage for current task
#         task_metrics_lists = {k: [] for k in all_stored_metrics}
        
#         # Temp storage for Sim Cap overlay
#         task_sim_epochs = []
#         task_sim_vals = [] 
        
#         for step in range(n_steps):
#             epoch_num = current_epoch + (step + 1) * config.log_frequency
            
#             # Temporary list for this step's repeats
#             step_res = {k: [] for k in all_stored_metrics}
            
#             # Run Simulated Capacity every 100 epochs
#             run_sim_cap = (epoch_num % 100 == 0)
            
#             for r in range(n_repeats):
#                 # Fold key for specific Step and Repeat to ensure uniqueness and reproducibility
#                 # Key structure: Master -> Task -> Step -> Repeat
#                 step_key = jax.random.fold_in(master_key, t_idx)
#                 step_key = jax.random.fold_in(step_key, step)
#                 unique_key = jax.random.fold_in(step_key, r)
                
#                 # Split key for separate sub-routines (GLUE vs SimCap)
#                 glue_key, sim_key = jax.random.split(unique_key)
                
#                 curr_reps = reps_data[step, r]
#                 curr_labels = labels_reshaped[:, r]
                
#                 # Check for NaNs in data
#                 if not jnp.all(jnp.isfinite(curr_reps)):
#                     for k in all_stored_metrics: step_res[k].append(jnp.nan)
#                     continue

#                 # Subsample data to keep QP solver efficient
#                 # We select 'config.analysis_subsamples' per class (using the config variable)
#                 unique_classes = jnp.unique(curr_labels)
#                 indices_list = []
#                 for cls in unique_classes:
#                     # Get all indices for this class
#                     cls_idxs = jnp.where(curr_labels == cls)[0]
                    
#                     # Randomly shuffle and pick top K
#                     # Use a subkey for permutation to ensure randomness
#                     perm_key = jax.random.fold_in(unique_key, int(cls))
#                     shuffled_cls_idxs = jax.random.permutation(perm_key, cls_idxs)
                    
#                     # Update: Use analysis_subsamples from config
#                     limit = min(len(cls_idxs), config.analysis_subsamples)
#                     indices_list.append(shuffled_cls_idxs[:limit])
                
#                 # Concatenate and filter data
#                 selected_idxs = jnp.concatenate(indices_list)
#                 curr_reps_sub = curr_reps[selected_idxs]
#                 curr_labels_sub = curr_labels[selected_idxs]

#                 try:
#                     # 1. Standard GLUE Metrics
#                     res = run_manifold_geometry(glue_key, curr_reps_sub, curr_labels_sub, n_samples_t=config.n_t)
#                     for k in glue_metrics: step_res[k].append(res[k])
                    
#                     # 2. Simulated Capacity & Accuracy
#                     if run_sim_cap:
#                         sc = compute_simulated_capacity(sim_key, curr_reps, curr_labels)
#                         glue_cap = res['Capacity']
                        
#                         # Calculate Relative Error
#                         if glue_cap > 1e-9:
#                             rel_error = abs(sc - glue_cap) / glue_cap
#                         else:
#                             rel_error = jnp.nan
                            
#                         step_res['Simulated_Capacity'].append(sc)
#                         step_res['Capacity_Relative_Error'].append(rel_error)
#                     else:
#                         step_res['Simulated_Capacity'].append(jnp.nan)
#                         step_res['Capacity_Relative_Error'].append(jnp.nan)

#                 except Exception as e:
#                     print(f"Error at step {step} rep {r}: {e}")
#                     for k in all_stored_metrics: step_res[k].append(jnp.nan)

#             # Store aggregated step data
#             for k in all_stored_metrics:
#                 task_metrics_lists[k].append(step_res[k])
                
#             # Logging & Overlay storage
#             if run_sim_cap:
#                 sim_caps = jnp.array(step_res['Simulated_Capacity'])
                
#                 # Note: We use jnp.nanmean
#                 task_sim_epochs.append(epoch_num)
#                 task_sim_vals.append(sim_caps)

#         # Finalize Task Data
#         full_results[t_name] = {}
#         for k in all_stored_metrics:
#             full_results[t_name][k] = jnp.array(task_metrics_lists[k])
            
#         # Add to plotting history (only GLUE metrics)
#         for k in glue_metrics:
#             plot_history[k].extend(task_metrics_lists[k])
            
#         # Process Sim Cap for overlay
#         if task_sim_epochs:
#             sim_vals_arr = jnp.array(task_sim_vals)
#             sim_means = jnp.nanmean(sim_vals_arr, axis=1)
#             sim_stds = jnp.nanstd(sim_vals_arr, axis=1)
#             sim_cap_history[t_name] = (task_sim_epochs, sim_means, sim_stds)

#         current_epoch += n_steps * config.log_frequency
#         task_boundaries.append(current_epoch)

#     # Convert plot history to simple arrays for matplotlib
#     # Note: We must convert JAX arrays to Numpy for Matplotlib
#     for k in plot_history:
#         if len(plot_history[k]) > 0:
#             plot_history[k] = jnp.array(plot_history[k]) 

#     # --- Plotting ---
#     if len(plot_history['Capacity']) == 0:
#         print("No manifold metrics computed.")
#         return full_results

#     n_metrics = len(glue_metrics)
#     fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3 * n_metrics), sharex=True)
#     if n_metrics == 1: axes = [axes]
    
#     total_steps = len(plot_history[glue_metrics[0]])
#     x_axis = jnp.arange(1, total_steps + 1) * config.log_frequency
    
#     colors = plt.cm.viridis(jnp.linspace(0, 0.9, n_metrics))
    
#     for i, metric in enumerate(glue_metrics):
#         ax = axes[i]
#         data = plot_history[metric]
        
#         if data.size == 0: continue
            
#         mean = jnp.nanmean(data, axis=1)
#         std = jnp.nanstd(data, axis=1)
        
#         color = colors[i]
#         ax.plot(x_axis, mean, label=f"GLUE {metric}", color=color, linewidth=2)
#         ax.fill_between(x_axis, mean - std, mean + std, color=color, alpha=0.2)
        
#         # --- Overlay Simulated Capacity ---
#         if metric == 'Capacity':
#             for t_name, (epochs, means, stds) in sim_cap_history.items():
#                 ax.errorbar(epochs, means, yerr=stds, fmt='o', color='black', 
#                             ecolor='gray', elinewidth=2, capsize=4, markersize=5,
#                             label='Simulated (Alg 1)' if t_name == task_names[0] else None)
#             ax.legend()
        
#         ax.set_ylabel(metric)
#         ax.grid(True, alpha=0.3)
        
#         for boundary in task_boundaries[:-1]:
#             ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
            
#         if i == 0:
#             ax.set_title(f"Manifold Geometric Analysis - {config.dataset_name}")

#     axes[-1].set_xlabel('Epochs')
#     plt.tight_layout()
#     save_path = os.path.join(config.figures_dir, f"manifold_metrics_{config.dataset_name}.png")
#     plt.savefig(save_path)
#     print(f"Manifold analysis plots saved to {save_path}")
    
#     return full_results