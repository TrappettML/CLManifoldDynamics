import os
import pickle
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, pearsonr

def load_algorithm_data(base_path):
    """
    Scans the base directory for algorithm subfolders, loading their 
    correlation_data.pkl if available.
    """
    data_by_algo = {}
    
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
        if metric.startswith('MTL_'):
            continue
            
        if metric.startswith('CL_'):
            categories['CL'].append(metric)
        elif metric.startswith('Glue_'):
            categories['GLUE'].append(metric)
        else:
            categories['Plasticity'].append(metric)
            
    return categories

def get_significance_asterisks(p_val):
    """Returns standard asterisk notation for p-values."""
    if p_val < 0.001:
        return '***'
    elif p_val < 0.01:
        return '**'
    elif p_val < 0.05:
        return '*'
    return 'ns'

def plot_grouped_boxplot(data_by_algo, category_name, metrics, save_path):
    """
    Generates a grouped boxplot comparing algorithms, runs Mann-Whitney U tests 
    with Bonferroni correction, saves stats to a file, and annotates the plot.
    """
    if not metrics:
        print(f"No metrics found for category: {category_name}. Skipping plot.")
        return

    algorithms = sorted(list(data_by_algo.keys()))
    num_algs = len(algorithms)
    num_metrics = len(metrics)
    
    fig_width = max(10, num_metrics * max(1.5, num_algs * 0.5))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    
    cmap = plt.get_cmap('tab10')
    algo_colors = {algo: cmap(i % 10) for i, algo in enumerate(algorithms)}
    
    total_group_width = 0.8
    box_width = total_group_width / num_algs
    legend_patches = []
    
    stats_file_path = save_path.replace('.png', '_stats.txt')
    
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write(f"Statistical Analysis: {category_name} Metrics\n")
        stats_file.write("="*60 + "\n")
        stats_file.write("Test: Pairwise Mann-Whitney U test with Bonferroni correction\n\n")
        
        for i, algo in enumerate(algorithms):
            color = algo_colors[algo]
            legend_patches.append(mpatches.Patch(color=color, label=algo))
            
        metric_max_y = {}
        box_positions = {}
        
        for j, metric in enumerate(metrics):
            box_positions[metric] = {}
            metric_data = []
            valid_algos = []
            
            for i, algo in enumerate(algorithms):
                pos = j + (i - num_algs / 2 + 0.5) * box_width
                
                if metric in data_by_algo[algo]:
                    vals = np.array(data_by_algo[algo][metric])
                    vals = vals[~np.isnan(vals)]
                    
                    if len(vals) > 0:
                        metric_data.append(vals)
                        valid_algos.append(algo)
                        box_positions[metric][algo] = pos
            
            if metric_data:
                bp = ax.boxplot(
                    metric_data, 
                    positions=list(box_positions[metric].values()), 
                    widths=box_width * 0.85, 
                    patch_artist=True, 
                    manage_ticks=False,
                    showfliers=True,
                    flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                    medianprops={'color': 'black', 'linewidth': 1.5}
                )
                
                for k, patch in enumerate(bp['boxes']):
                    algo_name = valid_algos[k]
                    color = algo_colors[algo_name]
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    bp['fliers'][k].set_markerfacecolor(color)
                
                all_vals = np.concatenate(metric_data)
                metric_max_y[metric] = np.max(all_vals)
                
                stats_file.write(f"\n--- Metric: {metric} ---\n")
                
                if len(valid_algos) < 2:
                    stats_file.write("Not enough algorithms with data to compare.\n")
                    continue
                
                comparisons = list(itertools.combinations(valid_algos, 2))
                num_comparisons = len(comparisons)
                
                stats_file.write(f"Number of comparisons (Bonferroni factor m): {num_comparisons}\n")
                stats_file.write(f"{'Comparison':<30} | {'U-stat':<10} | {'p-value':<12} | {'Adj p-value':<12} | {'Sig'}\n")
                stats_file.write("-" * 80 + "\n")
                
                significant_pairs = []
                
                for algo1, algo2 in comparisons:
                    data1 = np.array(data_by_algo[algo1][metric])
                    data2 = np.array(data_by_algo[algo2][metric])
                    
                    data1 = data1[~np.isnan(data1)]
                    data2 = data2[~np.isnan(data2)]
                    
                    try:
                        stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                        adj_p_val = min(1.0, p_val * num_comparisons)
                        asterisks = get_significance_asterisks(adj_p_val)
                        
                        stats_file.write(f"{algo1} vs {algo2:<15} | {stat:<10.2f} | {p_val:<12.2e} | {adj_p_val:<12.2e} | {asterisks}\n")
                        
                        if asterisks != 'ns':
                            significant_pairs.append({
                                'algos': (algo1, algo2),
                                'adj_p': adj_p_val,
                                'asterisks': asterisks
                            })
                    except ValueError as e:
                        stats_file.write(f"{algo1} vs {algo2:<15} | ERROR: {str(e)}\n")

                if significant_pairs:
                    significant_pairs.sort(key=lambda x: abs(box_positions[metric][x['algos'][0]] - box_positions[metric][x['algos'][1]]))
                    
                    y_base = metric_max_y[metric]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    step_h = max(y_range * 0.05, 0.05) 
                    
                    current_y = y_base + (step_h * 0.5) 
                    
                    for pair in significant_pairs:
                        x1 = box_positions[metric][pair['algos'][0]]
                        x2 = box_positions[metric][pair['algos'][1]]
                        
                        if x1 > x2:
                            x1, x2 = x2, x1
                        
                        ax.plot([x1, x1, x2, x2], [current_y, current_y + step_h*0.2, current_y + step_h*0.2, current_y], lw=1.2, color='black')
                        ax.text((x1 + x2) * 0.5, current_y + step_h*0.2, pair['asterisks'], ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')
                        current_y += step_h * 1.5

    ax.set_title(f"Algorithm Comparison: {category_name} Metrics", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Metric Value", fontsize=12)
    
    ax.set_xticks(np.arange(num_metrics))
    clean_labels = [m.replace('CL_', '').replace('Glue_', '') for m in metrics]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.15)
    
    for j in range(num_metrics - 1):
        ax.axvline(x=j + 0.5, color='grey', linestyle=':', alpha=0.4)
        
    ax.legend(handles=legend_patches, title="Algorithms", loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {category_name} comparison plot to {save_path}")

def compute_correlation_matrix(data_dict, metric_names):
    """Computes a Pearson correlation matrix for a given set of metrics."""
    n = len(metric_names)
    matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(metric_names):
        for j, name_j in enumerate(metric_names):
            if i == j:
                matrix[i, j] = 1.0
                continue
                
            vec_i = np.array(data_dict.get(name_i, []))
            vec_j = np.array(data_dict.get(name_j, []))
            
            mask = ~np.isnan(vec_i) & ~np.isnan(vec_j)
            
            if np.sum(mask) < 3:
                matrix[i, j] = np.nan
            else:
                matrix[i, j], _ = pearsonr(vec_i[mask], vec_j[mask])
                
    return matrix

def generate_difference_heatmaps(data_by_algo, output_dir):
    """
    Computes pairwise differences in correlation matrices between algorithms
    and saves them as heatmaps.
    """
    algorithms = sorted(list(data_by_algo.keys()))
    if len(algorithms) < 2:
        print("Not enough algorithms to compute correlation differences.")
        return

    comparisons = list(itertools.combinations(algorithms, 2))
    
    for algo1, algo2 in comparisons:
        # Find metrics common to both algorithms to ensure matrix alignment
        metrics1 = set(data_by_algo[algo1].keys())
        metrics2 = set(data_by_algo[algo2].keys())
        common_metrics = sorted(list(metrics1.intersection(metrics2)))
        
        if not common_metrics:
            print(f"No common metrics between {algo1} and {algo2}. Skipping heatmap.")
            continue
            
        corr_matrix1 = compute_correlation_matrix(data_by_algo[algo1], common_metrics)
        corr_matrix2 = compute_correlation_matrix(data_by_algo[algo2], common_metrics)
        
        diff_matrix = corr_matrix1 - corr_matrix2
        
        # Plotting the difference heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # vmin=-2 and vmax=2 because max difference between two [-1, 1] ranges is 2
        im = ax.imshow(diff_matrix, cmap='RdBu_r', vmin=-2, vmax=2)
        
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7)
        cbar.ax.set_ylabel(f"Difference in Pearson R ({algo1} - {algo2})", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(common_metrics)))
        ax.set_yticks(np.arange(len(common_metrics)))
        ax.set_xticklabels(common_metrics)
        ax.set_yticklabels(common_metrics)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        if len(common_metrics) < 20:
            for i in range(len(common_metrics)):
                for j in range(len(common_metrics)):
                    val = diff_matrix[i, j]
                    if not np.isnan(val):
                        # Highlight large differences in white for readability against dark colors
                        color = "white" if abs(val) > 1.0 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

        ax.set_title(f"Correlation Difference: {algo1} vs {algo2}", fontsize=16, fontweight='bold', pad=15)
        plt.tight_layout()
        
        filename = f"heatmap_diff_{algo1}_vs_{algo2}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"Saved correlation difference heatmap to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate algorithm comparison plots with statistical tests.")
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
        
    categories = categorize_metrics(data_by_algo)
    
    output_dir = os.path.join(base_path, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory ready: {output_dir}")
    
    print("\nGenerating Box Plots and Running Statistical Tests...")
    
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
    
    print("\nGenerating Correlation Difference Heatmaps...")
    generate_difference_heatmaps(data_by_algo, output_dir)
    
    print("\nAll plots and statistical reports generated successfully!")

if __name__ == "__main__":
    main()