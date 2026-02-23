import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_scatter(experiment_path, metric_x, metric_y, use_mtl=False):
    # 1. Locate Data
    filename = "mtl_correlation_data.pkl" if use_mtl else "correlation_data_no_plast.pkl"
    data_path = os.path.join(experiment_path, "correlations", filename)
    save_dir = os.path.join(experiment_path, "correlations")
    
    if not os.path.exists(data_path):
        print(f"[!] Data file not found: {data_path}")
        print("    Please run 'run_correlation_analysis.py' first.")
        return

    # 2. Load Data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 3. Validate Keys
    if metric_x not in data:
        print(f"[!] Metric X '{metric_x}' not found in data.")
        print(f"    Available keys: {list(data.keys())}")
        return
    if metric_y not in data:
        print(f"[!] Metric Y '{metric_y}' not found in data.")
        return

    x_data = data[metric_x]
    y_data = data[metric_y]

    # 4. Clean NaNs
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]

    if len(x_clean) < 3:
        print("[!] Not enough valid data points to plot.")
        return

    # 5. Compute Stats
    r_val, p_val = pearsonr(x_clean, y_clean)

    # 6. Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(x_clean, y_clean, alpha=0.7, edgecolors='w', s=80, color='royalblue')

    # Trend line
    if np.var(x_clean) > 0:
        m, b = np.polyfit(x_clean, y_clean, 1)
        plt.plot(x_clean, m*x_clean + b, color='darkorange', linestyle='--', linewidth=2, label=f'Fit (m={m:.2f})')

    plt.title(f"{metric_x} vs {metric_y}\nR = {r_val:.3f} (p={p_val:.3e})", fontsize=11)
    plt.xlabel(metric_x, fontsize=10)
    plt.ylabel(metric_y, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 7. Save
    prefix = "mtl_" if use_mtl else ""
    fname = f"{prefix}scatter_{metric_x}_vs_{metric_y}.png"
    fname = fname.replace(" ", "").replace("(", "").replace(")", "").replace("/", "_")
    save_path = os.path.join(save_dir, fname)
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"[x] Plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a scatter plot for two specific metrics.")
    parser.add_argument("exp_path", type=str, help="Path to the experiment folder")
    parser.add_argument("metric_x", type=str, help="Key name for the X-axis metric")
    parser.add_argument("metric_y", type=str, help="Key name for the Y-axis metric")
    parser.add_argument("--mtl", action="store_true", help="Use multi-task correlation data")
    
    args = parser.parse_args()
    
    if os.path.exists(args.exp_path):
        plot_scatter(args.exp_path, args.metric_x, args.metric_y, use_mtl=args.mtl)
    else:
        print(f"Experiment path not found: {args.exp_path}")