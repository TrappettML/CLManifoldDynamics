import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from typing import List, Dict, Tuple
from ipdb import set_trace

from glue_analyzer import GLUEAnalyzer

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
        # 1. FIX: Removed *100 from denominator for proper Xavier initialization
        self.W1 = np.random.randn(n_in, n_hid) / np.sqrt(n_in) 
        # 2. FIX: Added Bias for Hidden Layer
        self.b1 = np.zeros((1, n_hid))
        
        self.W2 = np.random.randn(n_hid, n_out) / np.sqrt(n_hid)
        # 2. FIX: Added Bias for Output Layer
        self.b2 = np.zeros((1, n_out))
        
        self.wd = weight_decay
        
    def forward(self, X):
        # 2. FIX: Added bias in forward pass
        self.H = np.tanh(X @ self.W1 + self.b1) 
        self.Z = self.H @ self.W2 + self.b2
        self.Out = 1 / (1 + np.exp(-self.Z))
        return self.Out, self.H
    
    def backward(self, X, Y, pred, lr=0.01):
        m = Y.shape[0]
        dZ = pred - Y
        
        # Gradients for W2 and b2
        dW2 = (self.H.T @ dZ) / m + self.wd * self.W2
        db2 = np.sum(dZ, axis=0, keepdims=True) / m  # Sum gradient for bias
        
        # Gradients for H (Hidden layer)
        dH = (dZ @ self.W2.T) * (1 - self.H**2) 
        
        # Gradients for W1 and b1
        dW1 = (X.T @ dH) / m + self.wd * self.W1
        db1 = np.sum(dH, axis=0, keepdims=True) / m  # Sum gradient for bias
        
        # Update weights and biases
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2 
        self.b2 -= lr * db2
        
        loss = -np.mean(Y * np.log(pred + 1e-8) + (1 - Y) * np.log(1 - pred + 1e-8))
        return loss
    
    def get_loss(self, X, Y, pred):
        loss = -np.mean(Y * np.log(pred + 1e-8) + (1 - Y) * np.log(1 - pred + 1e-8))
        return loss

class ThreeLayerNet:
    """
    3-Layer Network (Input -> H1 -> H2 -> Output)
    No Biases.
    Activations: ReLU, ReLU, Sigmoid
    """
    def __init__(self, n_in, n_h1, n_h2, n_out, weight_decay=1e-4):
        # He Initialization (Better for ReLU than Xavier)
        self.W1 = np.random.randn(n_in, n_h1) * np.sqrt(2 / n_in)
        self.W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2 / n_h1)
        self.W3 = np.random.randn(n_h2, n_out) * np.sqrt(2 / n_h2)
        
        self.wd = weight_decay
        
    def forward(self, X):
        # Layer 1: ReLU
        self.Z1 = X @ self.W1
        self.H1 = np.maximum(0, self.Z1) 
        
        # Layer 2: ReLU
        self.Z2 = self.H1 @ self.W2
        self.H2 = np.maximum(0, self.Z2)
        
        # Output Layer: Sigmoid
        self.Z3 = self.H2 @ self.W3
        self.Out = 1 / (1 + np.exp(-self.Z3))
        
        return self.Out, self.H2
    
    def backward(self, X, Y, pred, lr=0.01):
        m = Y.shape[0]
        Y = Y.reshape(-1, 1) # Safety fix
        
        # 1. Output Gradient (Sigmoid + BCE)
        dZ3 = pred - Y
        
        # 2. Gradients for W3
        dW3 = (self.H2.T @ dZ3) / m + self.wd * self.W3
        
        # 3. Backprop to Layer 2
        # Derivative of ReLU is 1 where x > 0, else 0
        dH2 = dZ3 @ self.W3.T
        dZ2 = dH2 * (self.Z2 > 0) 
        
        # 4. Gradients for W2
        dW2 = (self.H1.T @ dZ2) / m + self.wd * self.W2
        
        # 5. Backprop to Layer 1
        dH1 = dZ2 @ self.W2.T
        dZ1 = dH1 * (self.Z1 > 0)
        
        # 6. Gradients for W1
        dW1 = (X.T @ dZ1) / m + self.wd * self.W1
        
        # Update weights
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2
        self.W3 -= lr * dW3  
        
        loss = -np.mean(Y * np.log(pred + 1e-8) + (1 - Y) * np.log(1 - pred + 1e-8))
        return loss

    def get_loss(self, X, Y, pred):
        Y = Y.reshape(-1, 1)
        loss = -np.mean(Y * np.log(pred + 1e-8) + (1 - Y) * np.log(1 - pred + 1e-8))
        return loss


# --- Part 4: Experiment Loop ---
def run_experiment(config):
    X, Y = generate_xor_data(config.train_samples_per_cloud, config.noise_level)
    X_test, Y_test = generate_xor_data(config.test_samples_per_cloud, config.noise_level)
    # X = np.hstack([X_wo_bias, np.ones((X_wo_bias.shape[0], 1))])
    # model = TwoLayerNet(config.n_input, config.n_hidden, config.n_output, config.wd)
    model = ThreeLayerNet(config.n_input, config.n_h1, config.n_h2, config.n_output, config.wd)

    history = {'epoch': [], 'acc': [], 'loss': [], 'test_acc': [], 'test_loss': []}
    glue_metrics = {k: [] for k in ["Capacity", "Radius", "Dimension", "Center Align", "Axis Align", "Center-Axis"]}
    snapshots = {}
    
    print(f"Training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs + 1):
        pred, H = model.forward(X)
        loss = model.backward(X, Y, pred, config.learning_rate)
        acc = np.mean((pred > 0.5) == (Y > 0.5))

        if epoch % config.track_interval == 0:
            pred_test, H_test = model.forward(X_test)
            loss_test = model.get_loss(X_test, Y_test, pred_test)
            acc_test = np.mean((pred_test > 0.5) == (Y_test > 0.5))

            # save results
            history['test_acc'].append(acc_test)
            history['test_loss'].append(loss_test)
            history['epoch'].append(epoch)
            history['acc'].append(acc)
            history['loss'].append(loss)

            # save representations from test data samples
            idx0 = np.where(Y_test.flatten() == 0)[0]
            idx1 = np.where(Y_test.flatten() == 1)[0]
            manifolds = [H_test[idx0], H_test[idx1]]
            
            metrics = GLUEAnalyzer.analyze(manifolds, config.glue_n_t)

            for k, v in metrics.items():
                glue_metrics[k].append(v)
            
            if epoch % 500 == 0:
                print(f"Ep {epoch}: Loss {loss:.3f} | Acc {acc:.2f} | Cap {metrics['Capacity']:.3f} | R {metrics['Radius']:.2f} | D {metrics['Dimension']:.2f}")
            
            # Snapshots for plotting (Start, Early Learning, Converged)
            if epoch in [0, 400, 1000, 3000, config.epochs-1]:
                snapshots[epoch] = {
                    'H': H.copy(), 
                    'Y': Y.copy(),
                    'W1': model.W1.copy(), 
                    'W2': model.W2.copy(), 
                    'W3': model.W3.copy(),
                    # 'b1': model.b1.copy(),
                    # 'b2': model.b2.copy(),
                    'X': X.copy()
                }
                
    return history, glue_metrics, snapshots


def plot_results_3d(history, glue_metrics, snapshots):
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D


    snapshot_epochs = sorted(snapshots.keys())
    n_snaps = len(snapshot_epochs)
    
    # Reduced figure size: Width multiplier down to 3.5 (was 5.0), Height down to 10 (was 14)
    fig_width = max(n_snaps * 3.5, 12) 
    fig = plt.figure(figsize=(fig_width, 10))
    
    # ==========================================
    # ROW 1: METRICS (Static 2D)
    # ==========================================
    # We use a grid of 4 columns for the top row
    
    # 1. Performance
    ax_perf = fig.add_subplot(3, 4, 1)
    ax_perf.plot(history['epoch'], history['loss'], 'r--', lw=1.5, label='Train Loss')
    ax_perf.plot(history['epoch'], history['test_loss'], 'r-', lw=1.5, label='Test Loss')
    ax_perf.set_ylabel('Loss', color='r', fontsize=9)
    ax_perf.tick_params(axis='y', labelcolor='r', labelsize=8)
    
    ax_perf2 = ax_perf.twinx()
    ax_perf2.plot(history['epoch'], history['acc'], 'b--', lw=1.5, label='Train Acc')
    ax_perf2.plot(history['epoch'], history['test_acc'], 'b-', lw=1.5, label='Test Acc')
    ax_perf2.set_ylabel('Acc', color='b', fontsize=9)
    ax_perf2.tick_params(axis='y', labelcolor='b', labelsize=8)
    ax_perf.set_title("Performance", fontsize=10)
    ax_perf.legend(fontsize=8)
    ax_perf.grid(alpha=0.3)

    # 2. Capacity
    ax_cap = fig.add_subplot(3, 4, 2)
    ax_cap.plot(history['epoch'], glue_metrics['Capacity'], 'b-', lw=1.5)
    ax_cap.set_title("Capacity ($\\alpha_{glue}$)", fontsize=10)
    ax_cap.tick_params(labelsize=8)
    ax_cap.grid(alpha=0.3)

    # 3. Geometry
    ax_geo = fig.add_subplot(3, 4, 3)
    ax_geo.plot(history['epoch'], glue_metrics['Radius'], 'r-', lw=1.5, label='R')
    ax_geo.plot(history['epoch'], glue_metrics['Dimension'], 'g-', lw=1.5, label='D')
    ax_geo.set_title("Geometry", fontsize=10)
    ax_geo.legend(fontsize=8)
    ax_geo.tick_params(labelsize=8)
    ax_geo.grid(alpha=0.3)

    # 4. Alignment
    ax_align = fig.add_subplot(3, 4, 4)
    ax_align.plot(history['epoch'], glue_metrics['Center Align'], 'purple', lw=1.5, label=r'$\rho_c$')
    ax_align.plot(history['epoch'], glue_metrics['Axis Align'], 'orange', lw=1.5, label=r'$\rho_a$')
    ax_align.set_title("Alignment", fontsize=10)
    ax_align.legend(fontsize=8)
    ax_align.tick_params(labelsize=8)
    ax_align.grid(alpha=0.3)

    # ==========================================
    # LOOPS FOR SNAPSHOTS (ROWS 2 & 3)
    # ==========================================
    for i, ep in enumerate(snapshot_epochs):
        snap = snapshots[ep]
        H, Y = snap['H'], snap['Y']
        W1, b1, W2, b2 = snap['W1'], snap['b1'], snap['W2'], snap['b2']
        X_data = snap['X']
        idx0, idx1 = (Y.flatten() == 0), (Y.flatten() == 1)
        
        # --------------------------------------
        # ROW 2: HIDDEN GEOMETRY (Interactive 3D)
        # --------------------------------------
        # Note: We index into a grid of (3 rows, n_snaps columns) starting at index n_snaps + i + 1
        ax_h = fig.add_subplot(3, n_snaps, n_snaps + i + 1, projection='3d')
        
        # Raw Data (faint) - No labels needed
        ax_h.scatter(H[idx0,0], H[idx0,1], H[idx0,2], c='blue', alpha=0.1, s=5)
        ax_h.scatter(H[idx1,0], H[idx1,1], H[idx1,2], c='red', alpha=0.1, s=5)

        # Robust Geometry Calculation
        centers, sample_anchors, active_pts = GLUEAnalyzer.get_robust_geometry(H, Y)
        
        if centers is not None:
            # 1. Centers (Label only the first one to avoid duplicate legend entries)
            ax_h.scatter(centers[0,0], centers[0,1], centers[0,2], 
                        c='cyan', s=100, marker='*', edgecolors='black', label='Center')
            ax_h.scatter(centers[1,0], centers[1,1], centers[1,2], 
                        c='orange', s=100, marker='*', edgecolors='black') # No label
            
            # 2. Anchors (Label only the first one)
            ax_h.scatter(sample_anchors[0,0], sample_anchors[0,1], sample_anchors[0,2], 
                        c='cyan', s=40, marker='X', edgecolors='black', label='Anchor')
            ax_h.scatter(sample_anchors[1,0], sample_anchors[1,1], sample_anchors[1,2], 
                        c='orange', s=40, marker='X', edgecolors='black') # No label

            # 3. Axis Lines (Label only the first one)
            ax_h.plot([centers[0,0], sample_anchors[0,0]], 
                      [centers[0,1], sample_anchors[0,1]], 
                      [centers[0,2], sample_anchors[0,2]], 'k-', lw=1.5, label='Axis')
            ax_h.plot([centers[1,0], sample_anchors[1,0]], 
                      [centers[1,1], sample_anchors[1,1]], 
                      [centers[1,2], sample_anchors[1,2]], 'k-', lw=1.5) # No label

            # 4. Active Points (Label only if they exist)
            if active_pts[0].size > 0:
                ax_h.scatter(active_pts[0][:,0], active_pts[0][:,1], active_pts[0][:,2], 
                            c='blue', s=15, edgecolors='white', alpha=0.9, label='Support')
            if active_pts[1].size > 0:
                ax_h.scatter(active_pts[1][:,0], active_pts[1][:,1], active_pts[1][:,2], 
                            c='red', s=15, edgecolors='white', alpha=0.9) # No label

        # Decision Plane
        if W2.size == 3:
            w = W2.flatten()
            b = b2.item()
            tmp = np.linspace(-1.0, 1.0, 5)
            xx_p, yy_p = np.meshgrid(tmp, tmp)
            if abs(w[2]) > 1e-4:
                zz_p = (-w[0]*xx_p - w[1]*yy_p - b) / w[2]
                zz_p = np.clip(zz_p, -1.5, 1.5)
                # No label for plane to keep legend clean
                ax_h.plot_surface(xx_p, yy_p, zz_p, alpha=0.2, color='black')

        ax_h.set_title(f"Hidden (Ep {ep})", fontsize=10)
        
        # --- NEW: Axis Labels ---
        ax_h.set_xlabel('$h_1$', fontsize=8)
        ax_h.set_ylabel('$h_2$', fontsize=8)
        ax_h.set_zlabel('$h_3$', fontsize=8)
        
        # --- NEW: Legend ---
        # loc='lower left' usually interferes least with the 3D data in the center
        ax_h.legend(loc='lower left', fontsize=6, framealpha=0.5)

        ax_h.set_xlim(-1.1, 1.1); ax_h.set_ylim(-1.1, 1.1); ax_h.set_zlim(-1.1, 1.1)
        ax_h.view_init(elev=30, azim=45)
        ax_h.tick_params(axis='both', which='major', labelsize=6)

        # --------------------------------------
        # ROW 3: INPUT SPACE DECISION (2D)
        # --------------------------------------
        ax_in = fig.add_subplot(3, n_snaps, (2 * n_snaps) + i + 1)
        
        # Create grid for contour
        x_min, x_max, y_min, y_max = -2.5, 2.5, -2.5, 2.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Forward pass for background
        H_grid = np.tanh(grid @ W1 + b1)
        Z_grid = H_grid @ W2 + b2
        Prob = 1 / (1 + np.exp(-Z_grid))
        Prob = Prob.reshape(xx.shape)
        
        # Plot
        ax_in.contourf(xx, yy, Prob, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.2)
        ax_in.contour(xx, yy, Prob, levels=[0.5], colors='black', linewidths=1.5)
        ax_in.scatter(X_data[idx0, 0], X_data[idx0, 1], c='blue', alpha=0.6, s=5)
        ax_in.scatter(X_data[idx1, 0], X_data[idx1, 1], c='red', alpha=0.6, s=5)
        
        ax_in.set_title(f"Input (Ep {ep})", fontsize=10)
        ax_in.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"./xor_results.png")


def plot_results(history, glue_metrics, snapshots):
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Note: Removed Axes3D import as we are strictly 2D now

    snapshot_epochs = sorted(snapshots.keys())
    n_snaps = len(snapshot_epochs)
    
    fig_width = max(n_snaps * 3.5, 12) 
    fig = plt.figure(figsize=(fig_width, 10))
    
    # ==========================================
    # ROW 1: METRICS
    # ==========================================
    
    # 1. Performance
    ax_perf = fig.add_subplot(3, 4, 1)
    ax_perf.plot(history['epoch'], history['loss'], 'r--', lw=1.5, label='Train Loss')
    ax_perf.plot(history['epoch'], history['test_loss'], 'r-', lw=1.5, label='Test Loss')
    ax_perf.set_ylabel('Loss', color='r', fontsize=9)
    ax_perf.tick_params(axis='y', labelcolor='r', labelsize=8)
    
    ax_perf2 = ax_perf.twinx()
    ax_perf2.plot(history['epoch'], history['acc'], 'b--', lw=1.5, label='Train Acc')
    ax_perf2.plot(history['epoch'], history['test_acc'], 'b-', lw=1.5, label='Test Acc')
    ax_perf2.set_ylabel('Acc', color='b', fontsize=9)
    ax_perf2.tick_params(axis='y', labelcolor='b', labelsize=8)
    ax_perf.set_title("Performance", fontsize=10)
    ax_perf.legend(fontsize=8)
    ax_perf.grid(alpha=0.3)

    # 2. Capacity
    ax_cap = fig.add_subplot(3, 4, 2)
    ax_cap.plot(history['epoch'], glue_metrics['Capacity'], 'b-', lw=1.5)
    ax_cap.set_title("Capacity ($\\alpha_{glue}$)", fontsize=10)
    ax_cap.tick_params(labelsize=8)
    ax_cap.grid(alpha=0.3)

    # 3. Geometry
    ax_geo = fig.add_subplot(3, 4, 3)
    ax_geo.plot(history['epoch'], glue_metrics['Radius'], 'r-', lw=1.5, label='R')
    ax_geo.plot(history['epoch'], glue_metrics['Dimension'], 'g-', lw=1.5, label='D')
    ax_geo.set_title("Geometry", fontsize=10)
    ax_geo.legend(fontsize=8)
    ax_geo.tick_params(labelsize=8)
    ax_geo.grid(alpha=0.3)

    # 4. Alignment
    ax_align = fig.add_subplot(3, 4, 4)
    ax_align.plot(history['epoch'], glue_metrics['Center Align'], 'purple', lw=1.5, label=r'$\rho_c$')
    ax_align.plot(history['epoch'], glue_metrics['Axis Align'], 'orange', lw=1.5, label=r'$\rho_a$')
    ax_align.set_title("Alignment", fontsize=10)
    ax_align.legend(fontsize=8)
    ax_align.tick_params(labelsize=8)
    ax_align.grid(alpha=0.3)

    # ==========================================
    # LOOPS FOR SNAPSHOTS (ROWS 2 & 3)
    # ==========================================
    for i, ep in enumerate(snapshot_epochs):
        snap = snapshots[ep]
        H, Y = snap['H'], snap['Y'] # This is H2
        W1, W2, W3 = snap['W1'], snap['W2'], snap['W3']
        X_data = snap['X']
        idx0, idx1 = (Y.flatten() == 0), (Y.flatten() == 1)
        
        # --------------------------------------
        # ROW 2: LAST HIDDEN GEOMETRY (2D)
        # --------------------------------------
        ax_h = fig.add_subplot(3, n_snaps, n_snaps + i + 1)
        
        # Raw Data (faint)
        ax_h.scatter(H[idx0,0], H[idx0,1], c='blue', alpha=0.1, s=5)
        ax_h.scatter(H[idx1,0], H[idx1,1], c='red', alpha=0.1, s=5)

        # Robust Geometry Calculation
        centers, sample_anchors, active_pts = GLUEAnalyzer.get_robust_geometry(H, Y)
        
        if centers is not None:
            # 1. Centers
            ax_h.scatter(centers[0,0], centers[0,1], 
                        c='cyan', s=100, marker='*', edgecolors='black', label='Center')
            ax_h.scatter(centers[1,0], centers[1,1], 
                        c='orange', s=100, marker='*', edgecolors='black') 
            
            # 2. Anchors
            ax_h.scatter(sample_anchors[0,0], sample_anchors[0,1], 
                        c='cyan', s=40, marker='X', edgecolors='black', label='Anchor')
            ax_h.scatter(sample_anchors[1,0], sample_anchors[1,1], 
                        c='orange', s=40, marker='X', edgecolors='black') 

            # 3. Axis Lines
            ax_h.plot([centers[0,0], sample_anchors[0,0]], 
                      [centers[0,1], sample_anchors[0,1]], 'k-', lw=1.5, label='Axis')
            ax_h.plot([centers[1,0], sample_anchors[1,0]], 
                      [centers[1,1], sample_anchors[1,1]], 'k-', lw=1.5) 

            # 4. Active Points
            if active_pts[0].size > 0:
                ax_h.scatter(active_pts[0][:,0], active_pts[0][:,1], 
                            c='blue', s=15, edgecolors='white', alpha=0.9, label='Support')
            if active_pts[1].size > 0:
                ax_h.scatter(active_pts[1][:,0], active_pts[1][:,1], 
                            c='red', s=15, edgecolors='white', alpha=0.9)

        # Decision Boundary in H2
        # Equation: h2 @ W3 = 0 (since no bias)
        # h2_x * w3_x + h2_y * w3_y = 0 => y = -(w3_x / w3_y) * x
        w = W3.flatten()
        if abs(w[1]) > 1e-4:
            x_vals = np.linspace(-1.1, 1.1, 100)
            y_vals = -(w[0] / w[1]) * x_vals
            # Filter to plot limits
            mask = (y_vals >= -1.1) & (y_vals <= 1.1)
            ax_h.plot(x_vals[mask], y_vals[mask], 'k--', lw=2, label='Boundary')
        else:
            # Vertical line at x=0
            ax_h.vlines(0, -1.1, 1.1, colors='k', linestyles='--', lw=2, label='Boundary')

        ax_h.set_title(f"Hidden $H_2$ (Ep {ep})", fontsize=10)
        ax_h.set_xlabel('$h_{2,1}$', fontsize=8)
        ax_h.set_ylabel('$h_{2,2}$', fontsize=8)
        
        ax_h.legend(loc='lower right', fontsize=6, framealpha=0.5)

        ax_h.set_xlim(-1.1, 1.1)
        ax_h.set_ylim(-1.1, 1.1)
        ax_h.tick_params(axis='both', which='major', labelsize=6)
        ax_h.grid(True, alpha=0.2)

        # --------------------------------------
        # ROW 3: INPUT SPACE DECISION (2D)
        # --------------------------------------
        ax_in = fig.add_subplot(3, n_snaps, (2 * n_snaps) + i + 1)
        
        # Create grid for contour
        x_min, x_max, y_min, y_max = -2.5, 2.5, -2.5, 2.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Forward pass for background (3 Layers, No Bias)
        H1_grid = np.tanh(grid @ W1)
        H2_grid = np.tanh(H1_grid @ W2)
        Z_grid = H2_grid @ W3
        Prob = 1 / (1 + np.exp(-Z_grid))
        Prob = Prob.reshape(xx.shape)
        
        # Plot
        ax_in.contourf(xx, yy, Prob, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.2)
        ax_in.contour(xx, yy, Prob, levels=[0.5], colors='black', linewidths=1.5)
        ax_in.scatter(X_data[idx0, 0], X_data[idx0, 1], c='blue', alpha=0.6, s=5)
        ax_in.scatter(X_data[idx1, 0], X_data[idx1, 1], c='red', alpha=0.6, s=5)
        
        ax_in.set_title(f"Input (Ep {ep})", fontsize=10)
        ax_in.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(f"./xor_results_3layer.png")


if __name__=='__main__':
    # --- Configuration ---
    class Config:
        n_input: int = 2
        n_h1: int = 5
        n_h2: int = 2
        n_output: int = 1
        train_samples_per_cloud: int = 200
        test_samples_per_cloud: int = 100
        noise_level: float = 0.2
        learning_rate: float = 0.01
        epochs: int = 3000
        track_interval: int = 100
        
        # GLUE Analysis Parameters
        glue_n_t: int = 100  # n_t in Algorithm 2
        glue_epsilon: float = 1e-7
        wd: float = 1e-2

    cfg = Config()

    hist, metrics, snaps = run_experiment(cfg)
    # if cfg.n_hidden == 3 or getattr(cfg, 'n_hidden') == False:
    #     plot_results_3d(hist, metrics, snaps)
    plot_results(hist, metrics, snaps)