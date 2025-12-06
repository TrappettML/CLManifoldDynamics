import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Explicit import for 3d projection
from typing import List, Dict, Tuple
from glue_analyzer import GLUEAnalyzer
from ipdb import set_trace

# --- Part 1: Data Generation (XOR Problem) ---
def generate_xor_data(n_per_cloud: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates 4 noisy point clouds. 
    Class 0: (+1,+1), (-1,-1)
    Class 1: (+1,-1), (-1,+1)
    """
    means = np.array([
        [1, 1], [-1, -1],  # Class 0
        [1, -1], [-1, 1]   # Class 1
    ])
    
    X_list = []
    Y_list = []
    
    # Class 0
    for i in range(2):
        cloud = means[i] + np.random.randn(n_per_cloud, 2) * noise
        X_list.append(cloud)
        Y_list.append(np.zeros((n_per_cloud, 1)))
        
    # Class 1
    for i in range(2, 4):
        cloud = means[i] + np.random.randn(n_per_cloud, 2) * noise
        X_list.append(cloud)
        Y_list.append(np.ones((n_per_cloud, 1)))
        
    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y


class TwoLayerNet:
    def __init__(self, n_in, n_hid, n_out, weight_decay=1e-4):
        self.W1 = np.random.randn(n_in, n_hid) / np.sqrt(n_in) 
        self.b1 = np.zeros((1, n_hid))
        self.W2 = np.random.randn(n_hid, n_out) / np.sqrt(n_hid)
        self.b2 = np.zeros((1, n_out))
        self.wd = weight_decay
        
    def forward(self, X):
        self.H = np.tanh(X @ self.W1 + self.b1) 
        self.Z = self.H @ self.W2 + self.b2
        self.Out = 1 / (1 + np.exp(-self.Z))
        return self.Out, self.H
    
    def backward(self, X, Y, pred, lr=0.01):
        m = Y.shape[0]
        dZ = pred - Y
        
        dW2 = (self.H.T @ dZ) / m + self.wd * self.W2
        db2 = np.sum(dZ, axis=0, keepdims=True) / m
        dH = (dZ @ self.W2.T) * (1 - self.H**2) 
        dW1 = (X.T @ dH) / m + self.wd * self.W1
        db1 = np.sum(dH, axis=0, keepdims=True) / m
        
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2 
        self.b2 -= lr * db2
        
        loss = -np.mean(Y * np.log(pred + 1e-8) + (1 - Y) * np.log(1 - pred + 1e-8))
        return loss
    
    def get_loss(self, X, Y, pred):
        loss = -np.mean(Y * np.log(pred + 1e-8) + (1 - Y) * np.log(1 - pred + 1e-8))
        return loss


# --- Part 4: Experiment Loop ---
def run_experiment(config):
    model = TwoLayerNet(config.n_input, config.n_hidden, config.n_output, config.wd)

    history = {'epoch': [], 'acc': [], 'loss': [], 'test_acc': [], 'test_loss': []}
    glue_metrics = {k: [] for k in ["Capacity", "Radius", "Dimension", "Center Align", "Axis Align", "Center-Axis"]}
    snapshots = {}
    
    print(f"Training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs + 1):
        # generate new data every epoch to avoid minima
        X, Y = generate_xor_data(config.train_samples_per_cloud, config.noise_level)
        X_test, Y_test = generate_xor_data(config.test_samples_per_cloud, config.noise_level)
        
        pred, H = model.forward(X)
        loss = model.backward(X, Y, pred, config.learning_rate)
        acc = np.mean((pred > 0.5) == (Y > 0.5))

        if epoch % config.track_interval == 0:
            pred_test, H_test = model.forward(X_test)
            loss_test = model.get_loss(X_test, Y_test, pred_test)
            acc_test = np.mean((pred_test > 0.5) == (Y_test > 0.5))

            history['test_acc'].append(acc_test)
            history['test_loss'].append(loss_test)
            history['epoch'].append(epoch)
            history['acc'].append(acc)
            history['loss'].append(loss)

            # Separate classes for GLUE
            idx0 = np.where(Y_test.flatten() == 0)[0]
            idx1 = np.where(Y_test.flatten() == 1)[0]
            # Pass ALL test points to analyze, it will handle subsampling repeats
            manifolds = [H_test[idx0], H_test[idx1]]
            
            # --- UPDATED CALL ---
            metrics = GLUEAnalyzer.analyze(
                manifolds, 
                n_t_samples=config.glue_n_t,
                n_repeats=config.glue_repeats,     # New: Repeat measurement 10 times
                points_per_sample=config.glue_subsample # New: Sample 50 points per rep
            )

            for k, v in metrics.items():
                glue_metrics[k].append(v)
            
            if epoch % 500 == 0:
                print(f"Ep {epoch}: Loss {loss:.3f} | Acc {acc:.2f} | Cap {metrics['Capacity']:.3f} | R {metrics['Radius']:.2f} | D {metrics['Dimension']:.2f}")
            
            # Save Snapshots
            if epoch in [0, 200, 400, 600, config.epochs]:
                snapshots[epoch] = {
                    'H': H.copy(), 
                    'Y': Y.copy(),
                    'W1': model.W1.copy(), 
                    'b1': model.b1.copy(), 
                    'W2': model.W2.copy(), 
                    'b2': model.b2.copy(), 
                    'X': X.copy()
                }
                
    return history, glue_metrics, snapshots

def plot_metrics(plot_obj, history, glue_metrics, snapshots):
    # --- ROW 1: METRICS ---
    ax_perf = plot_obj.add_subplot(3, 4, 1)
    ax_perf.plot(history['epoch'], history['loss'], 'r--', label='Train Loss')
    ax_perf.plot(history['epoch'], history['test_loss'], 'r-', label='Test Loss')
    ax_perf2 = ax_perf.twinx()
    ax_perf2.plot(history['epoch'], history['acc'], 'b--', lw=1.5, label='Train Acc')
    ax_perf2.plot(history['epoch'], history['test_acc'], 'b-', lw=1.5, label='Test Acc')
    ax_perf2.set_ylabel('Acc', color='b')
    ax_perf.set_title("Performance")
    handles1, labels1 = ax_perf.get_legend_handles_labels()
    handles2, labels2 = ax_perf2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    ax_perf2.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(1.15, 1))

    
    ax_cap = plot_obj.add_subplot(3, 4, 2)
    ax_cap.plot(history['epoch'], glue_metrics['Capacity'], 'b-', label='Capactiy')
    ax_cap.legend()
    ax_cap.set_title("Capacity")

    ax_geo = plot_obj.add_subplot(3, 4, 3)
    ax_geo.plot(history['epoch'], glue_metrics['Radius'], 'r-', label='R')
    ax_geo.plot(history['epoch'], glue_metrics['Dimension'], 'g-', label='D')
    ax_geo.legend()
    ax_geo.set_title("Geometry")

    ax_align = plot_obj.add_subplot(3, 4, 4)
    ax_align.plot(history['epoch'], glue_metrics['Center Align'], 'purple', label=r'$\rho_c$')
    ax_align.plot(history['epoch'], glue_metrics['Axis Align'], 'orange', label=r'$\rho_a$')
    ax_align.legend()
    ax_align.set_title("Alignment")
    
    return plot_obj

def plot_results_3d(history, glue_metrics, snapshots):
    """ Plotting for 3D hidden space """
    snapshot_epochs = sorted(snapshots.keys())
    n_snaps = len(snapshot_epochs)
    
    fig_width = max(n_snaps * 3.5, 12) 
    fig = plt.figure(figsize=(fig_width, 10))
    
    fig = plot_metrics(fig, history, glue_metrics, snapshots)

    # --- ROW 2 & 3: SNAPSHOTS ---
    for i, ep in enumerate(snapshot_epochs):
        snap = snapshots[ep]
        H, Y = snap['H'], snap['Y']
        W2, b2 = snap['W2'], snap['b2']
        X_data = snap['X']
        W1, b1 = snap['W1'], snap['b1']
        
        idx0, idx1 = (Y.flatten() == 0), (Y.flatten() == 1)
        
        # --- ROW 2: 3D HIDDEN MANIFOLD ---
        ax_h = fig.add_subplot(3, n_snaps, n_snaps + i + 1, projection='3d')
        
        ax_h.scatter(H[idx0,0], H[idx0,1], H[idx0,2], c='blue', alpha=0.05, s=2)
        ax_h.scatter(H[idx1,0], H[idx1,1], H[idx1,2], c='red', alpha=0.05, s=2)

        # --- GLUE Geometry Visualization ---
        centers, anchor_cloud = GLUEAnalyzer.get_robust_geometry(H, Y)
        
        if centers is not None:
            # 1. Plot Centers (S0)
            ax_h.scatter(centers[:,0], centers[:,1], centers[:,2], 
                        c=['cyan','orange'], s=150, marker='*', edgecolors='k', label='Centers', zorder=10)
            
            # 2. Plot Anchor Cloud (S) and Lines (Axes)
            # anchor_cloud is a list of [sample_anchors_1, sample_anchors_2, ...]
            for sample_anchors in anchor_cloud:
                # Filter out NaNs for plotting lines (Matplotlib doesn't like NaNs in line plots)
                # Plot faint points for anchors
                ax_h.scatter(sample_anchors[:,0], sample_anchors[:,1], sample_anchors[:,2], 
                             c=['cyan','orange'], alpha=0.3, s=10)
                # Draw lines from Center to Anchors
                for m_idx in range(centers.shape[0]):
                    # Check for valid anchor
                    if np.isfinite(sample_anchors[m_idx]).all():
                        ax_h.plot([centers[m_idx, 0], sample_anchors[m_idx, 0]],
                                  [centers[m_idx, 1], sample_anchors[m_idx, 1]],
                                  [centers[m_idx, 2], sample_anchors[m_idx, 2]],
                                  color='black', alpha=0.4) # Increased alpha for visibility
            else:
                pass
        else:
            print(f"Center Anchor is None")

        # Draw Decision Plane
        w = W2.flatten()
        b = b2.flatten()[0]
        if abs(w[2]) > 1e-4:
            tmp = np.linspace(-1.0, 1.0, 10)
            xx_p, yy_p = np.meshgrid(tmp, tmp)
            zz_p = (-w[0]*xx_p - w[1]*yy_p - b) / w[2]
            zz_p = np.clip(zz_p, -1.5, 1.5)
            ax_h.plot_surface(xx_p, yy_p, zz_p, alpha=0.2, color='gray')

        ax_h.set_title(f"Ep {ep}", fontsize=10)
        ax_h.set_xlim(-1.1, 1.1); ax_h.set_ylim(-1.1, 1.1); ax_h.set_zlim(-1.1, 1.1)

        # --- ROW 3: INPUT SPACE ---
        ax_in = fig.add_subplot(3, n_snaps, (2 * n_snaps) + i + 1)
        x_min, x_max, y_min, y_max = -2.5, 2.5, -2.5, 2.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid = np.c_[xx.ravel(), yy.ravel()]
        H_grid = np.tanh(grid @ W1 + b1)
        Z_grid = H_grid @ W2 + b2
        Prob = 1 / (1 + np.exp(-Z_grid))
        Prob = Prob.reshape(xx.shape)
        
        ax_in.contourf(xx, yy, Prob, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.2)
        ax_in.contour(xx, yy, Prob, levels=[0.5], colors='black', linewidths=1.5)
        ax_in.scatter(X_data[idx0, 0], X_data[idx0, 1], c='blue', alpha=0.6, s=5)
        ax_in.scatter(X_data[idx1, 0], X_data[idx1, 1], c='red', alpha=0.6, s=5)
        ax_in.set_title(f"Input {ep}", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"./xor_results_3d_hidden.png")


def plot_results_2d(history, glue_metrics, snapshots):
    """ Plotting for 2D hidden space """
    snapshot_epochs = sorted(snapshots.keys())
    n_snaps = len(snapshot_epochs)
    fig_width = max(n_snaps * 3.5, 12) 
    fig = plt.figure(figsize=(fig_width, 10))
    
    # --- ROW 1: METRICS ---
    fig = plot_metrics(fig, history, glue_metrics, snapshots)

    # --- ROW 2 & 3: SNAPSHOTS ---
    for i, ep in enumerate(snapshot_epochs):
        snap = snapshots[ep]
        H, Y = snap['H'], snap['Y']
        W2, b2 = snap['W2'], snap['b2']
        X_data = snap['X']
        W1, b1 = snap['W1'], snap['b1']
        
        idx0, idx1 = (Y.flatten() == 0), (Y.flatten() == 1)
        
        # --- ROW 2: 2D HIDDEN MANIFOLD ---
        ax_h = fig.add_subplot(3, n_snaps, n_snaps + i + 1)
        ax_h.scatter(H[idx0,0], H[idx0,1], c='blue', alpha=0.1, s=5)
        ax_h.scatter(H[idx1,0], H[idx1,1], c='red', alpha=0.1, s=5)

        # --- GLUE Geometry Visualization ---
        centers, anchor_cloud = GLUEAnalyzer.get_robust_geometry(H, Y)
        
        if centers is not None:
            # 1. Plot Centers (Star)
            ax_h.scatter(centers[:,0], centers[:,1], c=['cyan','orange'], s=200, marker='*', edgecolors='k', zorder=10)
            
            # 2. Plot Anchor Cloud (Axes visualization)
            if len(anchor_cloud) == 0:
                print("Anchor Cloud is Empy")
            for sample_anchors in anchor_cloud:
                # Plot faint anchor points
                ax_h.scatter(sample_anchors[:,0], sample_anchors[:,1], c=['cyan','orange'], alpha=0.4, s=15)
                # Draw faint lines from Center to Anchors
                for m_idx in range(centers.shape[0]):
                    # Explicit Check for NaNs
                    if np.isfinite(sample_anchors[m_idx]).all():
                        ax_h.plot([centers[m_idx, 0], sample_anchors[m_idx, 0]], 
                                  [centers[m_idx, 1], sample_anchors[m_idx, 1]], 
                                  color='black', alpha=0.3) # Increased Alpha

        # Decision Boundary
        w = W2.flatten()
        b = b2.flatten()[0]
        if abs(w[1]) > 1e-4:
            x_vals = np.linspace(-1.1, 1.1, 100)
            y_vals = -(w[0] * x_vals + b) / w[1]
            mask = (y_vals >= -1.1) & (y_vals <= 1.1)
            ax_h.plot(x_vals[mask], y_vals[mask], 'k--', lw=2)
        else:
            ax_h.vlines(-b/w[0], -1.1, 1.1, colors='k', linestyles='--')

        ax_h.set_title(f"Hidden Ep {ep}")
        ax_h.set_xlim(-1.1, 1.1); ax_h.set_ylim(-1.1, 1.1)

        # --- ROW 3: INPUT SPACE ---
        ax_in = fig.add_subplot(3, n_snaps, (2 * n_snaps) + i + 1)
        x_min, x_max, y_min, y_max = -2.5, 2.5, -2.5, 2.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid = np.c_[xx.ravel(), yy.ravel()]
        H_grid = np.tanh(grid @ W1 + b1)
        Z_grid = H_grid @ W2 + b2
        Prob = 1 / (1 + np.exp(-Z_grid))
        Prob = Prob.reshape(xx.shape)
        
        ax_in.contourf(xx, yy, Prob, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.2)
        ax_in.contour(xx, yy, Prob, levels=[0.5], colors='black', linewidths=1.5)
        ax_in.scatter(X_data[idx0, 0], X_data[idx0, 1], c='blue', alpha=0.6, s=5)
        ax_in.scatter(X_data[idx1, 0], X_data[idx1, 1], c='red', alpha=0.6, s=5)
        ax_in.set_title(f"Input {ep}")

    plt.tight_layout()
    plt.savefig(f"./xor_results_2d_hidden.png")


if __name__=='__main__':
    class Config:
        n_input: int = 2
        n_hidden: int = 2
        n_output: int = 1
        train_samples_per_cloud: int = 200
        test_samples_per_cloud: int = 200 # Provided large set for subsampling
        noise_level: float = 0.2
        learning_rate: float = 0.1
        epochs: int = 1000
        track_interval: int = 100
        
        # GLUE Analysis Parameters (Section E.1.4)
        glue_n_t: int = 100       # t samples per repeat
        glue_repeats: int = 10    # statistical repeats
        glue_subsample: int = 50  # points per manifold per repeat
        
        wd: float = 1e-4

    cfg = Config()

    hist, metrics, snaps = run_experiment(cfg)
    if cfg.n_hidden == 3:
        print("Plotting 3D Hidden Manifold...")
        plot_results_3d(hist, metrics, snaps)
    else:
        print("Plotting 2D Hidden Manifold...")
        plot_results_2d(hist, metrics, snaps)