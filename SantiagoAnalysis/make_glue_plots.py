import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import mannwhitneyu
from itertools import combinations

# ==========================================
# 1. Configuration and Setup
# ==========================================
stim_types = ['AM', 'PT', 'NS']
periods = ['base', 'onset', 'sustained', 'offset']
regions = [
    'Primary auditory area',
    'Ventral auditory area',
    'Dorsal auditory area',
    'Posterior auditory area'
]

region_label_map = {
    'Primary auditory area': 'Pri',
    'Ventral auditory area': 'Ven',
    'Dorsal auditory area': 'Dor',
    'Posterior auditory area': 'Pos',
    'Dor+Pos auditory area': 'Dor+Pos'
}

glue_metrics = [
    'capacity', 
    'dimension', 
    'radius', 
    'center_align', 
    'axis_align', 
    'center_axis_align', 
    'approx_capacity'
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Output directories
os.makedirs('figs', exist_ok=True)
STATS_FILENAME = 'figs/statistical_results.txt'

# ==========================================
# 2. Data Loading and Pivoting
# ==========================================

def load_and_pivot_data(filename):
    print(f"Loading {filename}...", flush=True)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    raw_means = data['means']
    pivoted_means = {}
    n_metrics = 0
    
    for key, value_array in raw_means.items():
        session, stim, region, period = key
        
        if n_metrics == 0:
            n_metrics = value_array.shape[0] if hasattr(value_array, 'shape') else len(value_array)
            
        pivot_key = (period, stim, region)
        
        if pivot_key not in pivoted_means:
            pivoted_means[pivot_key] = []
            
        pivoted_means[pivot_key].append(value_array)
        
    print(f"Data loaded. Found {n_metrics} metric(s) across {len(raw_means)} conditions.")
    return pivoted_means, n_metrics

# ==========================================
# 3. Statistical Helpers
# ==========================================

def get_star_string(p):
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'

def compute_pairwise_stats(data_map, regions, alpha=0.05):
    """
    Computes pairwise Mann-Whitney U tests with Bonferroni correction.
    
    Args:
        data_map: dict {region_name: [values]}
        regions: list of region names to compare
        
    Returns:
        significant_pairs: list of dicts with keys (r1, r2, p_adj, p_raw, u_stat, r1_idx, r2_idx)
    """
    n_regions = len(regions)
    if n_regions < 2:
        return []

    # Bonferroni correction factor: number of unique pairs
    n_comparisons = (n_regions * (n_regions - 1)) / 2
    
    results = []
    
    # Iterate over all unique pairs
    for (i, r1), (j, r2) in combinations(enumerate(regions), 2):
        vals1 = data_map.get(r1, [])
        vals2 = data_map.get(r2, [])
        
        # Only test if we have enough data
        if len(vals1) > 0 and len(vals2) > 0:
            try:
                u_stat, p_val = mannwhitneyu(vals1, vals2, alternative='two-sided')
                p_adj = min(p_val * n_comparisons, 1.0)
                
                if p_adj < alpha:
                    results.append({
                        'r1': r1,
                        'r2': r2,
                        'r1_idx': i,
                        'r2_idx': j,
                        'p_raw': p_val,
                        'p_adj': p_adj,
                        'u_stat': u_stat,
                        'star': get_star_string(p_adj)
                    })
            except ValueError:
                # Occurs if all numbers are identical in both sets
                continue
                
    return results

def annotate_stats(ax, sig_pairs, x_positions, base_y_height, y_step):
    """
    Draws brackets and stars for significant pairs.
    """
    # Sort pairs by distance between bars (smallest distance first) to stack neatly
    sig_pairs.sort(key=lambda x: abs(x['r1_idx'] - x['r2_idx']))
    
    current_y = base_y_height
    
    for res in sig_pairs:
        x1 = x_positions[res['r1_idx']]
        x2 = x_positions[res['r2_idx']]
        
        # Ensure x1 is left of x2
        if x1 > x2: x1, x2 = x2, x1
            
        # Draw bracket
        bar_h = y_step * 0.2
        ax.plot([x1, x1, x2, x2], [current_y, current_y + bar_h, current_y + bar_h, current_y], lw=1, c='k')
        
        # Draw star
        ax.text((x1 + x2) * 0.5, current_y + bar_h, res['star'], 
                ha='center', va='bottom', color='k', fontsize=10)
        
        # Increment Y for next bracket
        current_y += y_step

    return current_y # Return new max height

# ==========================================
# 4. Plotting Functions
# ==========================================

def _extract_metric_list(values_list, metric_index):
    return [v[metric_index] for v in values_list]

def grouped_positions(n_groups, group_size, gap=1, start=1):
    positions = []
    pos = start
    for _ in range(n_groups):
        for _ in range(group_size):
            positions.append(pos)
            pos += 1
        pos += gap
    return positions

def plot_metric_boxplot(results_dict, period, stim_types, regions, metric_index,
                        stats_file, metric_name,
                        ax=None, title=None, ylabel=None, colors=None, region_label_map=None):
    
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # 1. Collect Data for Plotting
    all_plot_data = []
    all_labels = []
    
    # Store data separated by stim type for statistical processing
    # Structure: stim_data_groups[stim_index] = {region_name: values}
    stim_data_groups = []

    for stim_type in stim_types:
        current_stim_map = {}
        for region in regions:
            key = (period, stim_type, region)
            
            if key in results_dict and results_dict[key]:
                vals = _extract_metric_list(results_dict[key], metric_index)
                current_stim_map[region] = vals
                all_plot_data.append(vals)
                
                lbl = region_label_map.get(region, region[:3]) if region_label_map else region[:3]
                all_labels.append(f"{stim_type} - {lbl}")
            else:
                current_stim_map[region] = []
                all_plot_data.append([])
                all_labels.append(f"{stim_type} - N/A")
        
        stim_data_groups.append(current_stim_map)

    # 2. Plot Boxplots
    positions = grouped_positions(len(stim_types), len(regions), gap=1)
    
    # Filter empty for plotting to prevent errors
    valid_data = [d for d in all_plot_data if len(d) > 0]
    valid_positions = [p for d, p in zip(all_plot_data, positions) if len(d) > 0]

    if valid_data:
        boxes = ax.boxplot(valid_data, positions=valid_positions, patch_artist=True,
                           medianprops=dict(color='black'), showfliers=False)
        
        for i, box in enumerate(boxes['boxes']):
            box.set_facecolor(colors[i % len(regions)])

        # Calculate Y-limit to prepare for annotations
        # Flatten data to find max
        flat_vals = [item for sublist in valid_data for item in sublist]
        if flat_vals:
            max_val = max(flat_vals)
            min_val = min(flat_vals)
            y_range = max_val - min_val
            y_step = y_range * 0.1 if y_range > 0 else 1.0
            current_max_y = max_val + (y_step * 0.5)
        else:
            current_max_y = 1
            y_step = 0.1
            
    else:
        current_max_y = 1
        y_step = 0.1

    # 3. Perform Stats and Annotate
    
    # Write Header to stats file
    stats_file.write(f"\n--- {metric_name} | {period.upper()} ---\n")
    
    for i, stim_type in enumerate(stim_types):
        # Identify the x-positions for this specific group
        # positions list is flat. If 4 regions, group 0 is indices 0-3, group 1 is 4-7, etc.
        group_start_idx = i * len(regions) 
        group_positions = positions[group_start_idx : group_start_idx + len(regions)]
        
        # Compute stats
        sig_pairs = compute_pairwise_stats(stim_data_groups[i], regions)
        
        # Write to file
        if sig_pairs:
            stats_file.write(f"  Stimulus: {stim_type}\n")
            for res in sig_pairs:
                stats_file.write(f"    {res['r1']} vs {res['r2']}: "
                                 f"p_adj={res['p_adj']:.4e} ({res['star']})\n")
            
            # Draw annotations
            # We pass the subset of positions corresponding to this group
            new_max = annotate_stats(ax, sig_pairs, group_positions, current_max_y, y_step)
            # Update max y to ensure we don't overwrite if groups are close (though here they are separated by x)
            # For purely visual consistency, we might want to keep the highest Y across all groups
            if new_max > current_max_y:
                current_max_y = new_max
        else:
             stats_file.write(f"  Stimulus: {stim_type} - No significant differences.\n")

    # Adjust plot limits to fit annotations
    ax.set_ylim(top=current_max_y + y_step)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# ==========================================
# 5. Main Execution
# ==========================================

def main():
    filename = './analysis_results.pkl'
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    # 1. Load Data
    pivoted_data, n_metrics = load_and_pivot_data(filename)
    
    # Open stats file
    print(f"Opening {STATS_FILENAME} for logging statistics...")
    with open(STATS_FILENAME, 'w') as f_stats:
        f_stats.write("Statistical Analysis Results (Mann-Whitney U, Bonferroni Corrected)\n")
        f_stats.write("===================================================================\n")
    
        # 2. Iterate over every metric
        for m in range(n_metrics):
            
            if m < len(glue_metrics):
                raw_name = glue_metrics[m]
                display_name = raw_name.replace('_', ' ').title()
                file_name_slug = raw_name
            else:
                display_name = f"Metric {m}"
                file_name_slug = f"metric_{m}"

            print(f"Generating plots for {display_name}...")
            
            # 3. Iterate over periods
            for period in periods:
                plt.figure(figsize=(10, 6)) # Slightly wider to accommodate labels
                
                plot_metric_boxplot(
                    pivoted_data, 
                    period, 
                    stim_types, 
                    regions,
                    metric_index=m,
                    stats_file=f_stats,      # Pass file handle
                    metric_name=display_name, # Pass name for logging
                    title=f'{display_name} - {period.capitalize()} Period',
                    ylabel=f'{display_name}',
                    colors=colors,
                    region_label_map=region_label_map
                )
                
                plt.tight_layout()
                savename = f'figs/{file_name_slug}_summary_{period}.pdf'
                plt.savefig(savename)
                plt.close()

    print(f"\nProcessing complete. Statistical results saved to {STATS_FILENAME}")

if __name__ == "__main__":
    main()