import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Configuration and Setup
# ==========================================
# Parameters matching your main script
stim_types = ['AM', 'PT', 'NS']
periods = ['base', 'onset', 'sustained', 'offset']
regions = [
    'Primary auditory area',
    'Ventral auditory area',
    'Dorsal auditory area',
    'Posterior auditory area'
]

# Short names for x-axis labels
region_label_map = {
    'Primary auditory area': 'Pri',
    'Ventral auditory area': 'Ven',
    'Dorsal auditory area': 'Dor',
    'Posterior auditory area': 'Pos',
    'Dor+Pos auditory area': 'Dor+Pos'
}

# Colors for the boxplots (one per region)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Ensure output directory exists
os.makedirs('figs', exist_ok=True)

# ==========================================
# 2. Data Loading and Pivoting
# ==========================================

def load_and_pivot_data(filename):
    """
    Loads the pickle file and reorganizes data from:
      dict[(session, stim, region, period)] = value_array
    to:
      dict[(period, stim, region)] = list_of_values_across_sessions
    
    Returns:
        pivoted_means: Dict of lists for the means
        n_metrics: The number of metrics found in the value arrays
    """
    print(f"Loading {filename}...", flush=True)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    raw_means = data['means']
    # raw_sems = data['sems'] # Not used for boxplots across sessions
    
    pivoted_means = {}
    n_metrics = 0
    
    # Iterate through the raw dictionary to pivot
    for key, value_array in raw_means.items():
        session, stim, region, period = key
        
        # Determine number of metrics (dimensions) from the first entry found
        if n_metrics == 0:
            n_metrics = value_array.shape[0] if hasattr(value_array, 'shape') else len(value_array)
            
        # Create the new key expected by the plotting logic
        pivot_key = (period, stim, region)
        
        if pivot_key not in pivoted_means:
            pivoted_means[pivot_key] = []
            
        pivoted_means[pivot_key].append(value_array)
        
    print(f"Data loaded. Found {n_metrics} metric(s) across {len(raw_means)} conditions.")
    return pivoted_means, n_metrics

# ==========================================
# 3. Plotting Functions (Adapted from analysis.py)
# ==========================================

def _extract_metric_list(values_list, metric_index):
    """Extracts the specific scalar metric from the list of arrays."""
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

def collect_boxplot_data(results_dict, period, stim_types, regions, metric_index, region_label_map=None):
    data, labels = [], []
    for stim_type in stim_types:
        for region in regions:
            key = (period, stim_type, region)
            
            # Check if we have data for this condition
            if key in results_dict and results_dict[key]:
                # Extract the specific metric (e.g., metric 0) from the arrays
                vals = _extract_metric_list(results_dict[key], metric_index)
                data.append(vals)
                
                # Create label
                if region_label_map:
                    labels.append(f"{stim_type} - {region_label_map.get(region, region[:3])}")
                else:
                    labels.append(f"{stim_type} - {region[:3]}")
            else:
                # Handle missing data gracefully (placeholder)
                data.append([])
                labels.append(f"{stim_type} - N/A")
                
    return data, labels

def plot_metric_boxplot(results_dict, period, stim_types, regions, metric_index,
                        ax=None, title=None, ylabel=None, colors=None, region_label_map=None):
    
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    data, labels = collect_boxplot_data(
        results_dict, period, stim_types, regions,
        metric_index=metric_index,
        region_label_map=region_label_map
    )
    
    positions = grouped_positions(len(stim_types), len(regions), gap=1)

    # Filter out empty data lists to avoid plotting errors
    valid_data = [d for d in data if len(d) > 0]
    valid_positions = [p for d, p in zip(data, positions) if len(d) > 0]

    if valid_data:
        boxes = ax.boxplot(valid_data, positions=valid_positions, patch_artist=True,
                           medianprops=dict(color='black'))
        
        # Color the boxes cyclically based on regions
        for i, box in enumerate(boxes['boxes']):
            box.set_facecolor(colors[i % len(regions)])
            
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    filename = './analysis_results.pkl'
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    # 1. Load Data
    pivoted_data, n_metrics = load_and_pivot_data(filename)
    
    # 2. Iterate over every metric found in the results
    for m in range(n_metrics):
        print(f"\nGenerating plots for Metric {m}...")
        
        # 3. Iterate over every time period
        for period in periods:
            plt.figure(figsize=(8, 5)) # Adjusted size for better visibility
            
            plot_metric_boxplot(
                pivoted_data, 
                period, 
                stim_types, 
                regions,
                metric_index=m,
                title=f'Metric {m} - {period.capitalize()} Period',
                ylabel=f'Metric {m} Value',
                colors=colors,
                region_label_map=region_label_map
            )
            
            plt.tight_layout()
            
            # Save figure
            savename = f'figs/metric_{m}_summary_{period}.pdf'
            plt.savefig(savename)
            print(f"Saved {savename}")
            plt.close()
            # plt.show()

if __name__ == "__main__":
    main()