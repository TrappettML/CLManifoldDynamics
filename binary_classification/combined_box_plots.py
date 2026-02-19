import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_algorithm_data(base_path):
    """
    Scans the base directory for algorithm subfolders, loading their 
    correlation_data.pkl if available.
    """
    data_by_algo = {}
    
    # Iterate through immediate subdirectories (these should be the algorithms)
    for entry in os.listdir(base_path):
        algo_dir = os.path.join(base_path, entry)
        if not os.path.isdir(algo_dir):
            continue
            
        corr_file = os.path.join(algo_dir, "correlations", "correlation_data.pkl")
        
        if os.path.exists(corr_file):
            with open(corr_file, 'rb') as f:
                data = pickle.load(f)
                data_by_algo[entry] = data
                print(f"Loaded data for algorithm: {entry} ({len(data)} metrics found)")
        else:
            print(f"Skipping {entry}: No correlation_data.pkl found.")
            
    return data_by_algo

def categorize_metrics(data_by_algo):
    """
    Scans all loaded data to find the unique metrics and groups them 
    into CL, GLUE, and Plasticity categories.
    """
    all_metrics = set()
    for algo_data in data_by_algo.values():
        all_metrics.update(algo_data.keys())
        
    categories = {
        'CL': [],
        'GLUE': [],
        'Plasticity': []
    }
    
    for metric in sorted(list(all_metrics)):
        # Skip MTL baselines to focus purely on the algorithm comparison
        if metric.startswith('MTL_'):
            continue
            
        if metric.startswith('CL_'):
            categories['CL'].append(metric)
        elif metric.startswith('Glue_'):
            categories['GLUE'].append(metric)
        else:
            # Everything else (e.g., ActiveUnits_t1t0, WeightMag, etc.) 
            # falls into Plasticity based on the correlation generation logic
            categories['Plasticity'].append(metric)
            
    return categories

def plot_grouped_boxplot(data_by_algo, category_name, metrics, save_path):
    """
    Generates and saves a grouped boxplot comparing algorithms across given metrics.
    """
    if not metrics:
        print(f"No metrics found for category: {category_name}. Skipping plot.")
        return

    algorithms = sorted(list(data_by_algo.keys()))
    num_algs = len(algorithms)
    num_metrics = len(metrics)
    
    # Dynamically size the figure based on the number of metrics so it doesn't get cramped
    fig_width = max(10, num_metrics * max(1.2, num_algs * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    
    # Colors for the algorithms
    cmap = plt.get_cmap('tab10')
    algo_colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    # Width settings for grouped boxes
    total_group_width = 0.8
    box_width = total_group_width / num_algs
    
    legend_patches = []
    
    for i, algo in enumerate(algorithms):
        color = algo_colors[algo]
        legend_patches.append(mpatches.Patch(color=color, label=algo))
        
        plot_data = []
        plot_positions = []
        
        for j, metric in enumerate(metrics):
            # Calculate position for this specific algorithm's box within the metric group
            pos = j + (i - num_algs / 2 + 0.5) * box_width
            
            if metric in data_by_algo[algo]:
                # Extract and clean data (remove NaNs)
                vals = np.array(data_by_algo[algo][metric])
                vals = vals[~np.isnan(vals)]
                
                if len(vals) > 0:
                    plot_data.append(vals)
                    plot_positions.append(pos)
        
        # Plot boxes for this algorithm if data exists
        if plot_data:
            bp = ax.boxplot(
                plot_data, 
                positions=plot_positions, 
                widths=box_width * 0.85, # Slight gap between boxes in a group
                patch_artist=True, 
                manage_ticks=False,
                showfliers=True,
                flierprops={'marker': 'o', 'markerfacecolor': color, 'markersize': 4, 'alpha': 0.5},
                medianprops={'color': 'black', 'linewidth': 1.5}
            )
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

    # Formatting best practices
    ax.set_title(f"Algorithm Comparison: {category_name} Metrics", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Metric Value (Relative Difference / Score)", fontsize=12)
    
    # X-axis ticks centered on the metric groups
    ax.set_xticks(np.arange(num_metrics))
    # Clean up metric names for the labels (remove repetitive prefixes)
    clean_labels = [m.replace('CL_', '').replace('Glue_', '') for m in metrics]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add vertical dividers to clearly separate metric groups
    for j in range(num_metrics - 1):
        ax.axvline(x=j + 0.5, color='grey', linestyle=':', alpha=0.4)
        
    ax.legend(handles=legend_patches, title="Algorithms", loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {category_name} comparison plot to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate algorithm comparison grouped box plots.")
    parser.add_argument(
        '--experiment_folder', 
        type=str, 
        required=True,
        help="Path to the dataset experiment folder (e.g., /.../results/mnist/)"
    )
    args = parser.parse_args()
    
    base_path = args.experiment_folder
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Experiment folder not found: {base_path}")
        
    print(f"\nScanning for algorithms in: {base_path}")
    data_by_algo = load_algorithm_data(base_path)
    
    if not data_by_algo:
        print("No valid correlation data found. Exiting.")
        return
        
    # Group the metrics
    categories = categorize_metrics(data_by_algo)
    
    # Setup Output Directory
    output_dir = os.path.join(base_path, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory ready: {output_dir}")
    
    # Generate Plots
    print("\nGenerating Plots...")
    
    plot_grouped_boxplot(
        data_by_algo, 
        category_name="Continual Learning (CL)", 
        metrics=categories['CL'], 
        save_path=os.path.join(output_dir, "cl_metrics_comparison.png")
    )
    
    plot_grouped_boxplot(
        data_by_algo, 
        category_name="Manifold Geometry (GLUE)", 
        metrics=categories['GLUE'], 
        save_path=os.path.join(output_dir, "glue_metrics_comparison.png")
    )
    
    plot_grouped_boxplot(
        data_by_algo, 
        category_name="Network Plasticity", 
        metrics=categories['Plasticity'], 
        save_path=os.path.join(output_dir, "plasticity_metrics_comparison.png")
    )
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()