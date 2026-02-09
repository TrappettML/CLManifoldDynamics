import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu, ks_2samp
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


path = '/home/users/MTrappett/manifold/SantiagoAnalysis/jaramillo_data/'
stim_types = ['AM', 'PT', 'NS']
periods = ['base', 'onset', 'sustained', 'offset']
regions = ['Primary auditory area',
           'Ventral auditory area',
           'Dorsal auditory area',
           'Posterior auditory area']


def load_data(stim_type, pool_dorsal_posterior=False):
    '''
    Load the saved neural and behavioral data as a dictionary.

    Parameters
    --
    stim_type : Type of stimulus. Should be one of ['AM', 'PT', 'NS'].

    Returns
    --
    data : A dictionary with the following keys and values...
        - base : array of shape (n_trials, n_units) with pre-stimulus spike counts.
        - onset : array of shape (n_trials, n_units) with spike counts following stimulus onset.
        - sustained : array of shape (n_trials, n_units) with spike counts during stimulus.
        - offset : array of shape (n_trials, n_units) with spike counts following stimulus offset.
        - brainRegionArray : array of shape (n_units,) with brain regions for each unit.
        - mouseIDArray : array of shape (n_units,) with ID of mouse from which each unit was recorded.
        - sessionIDArray : array of shape (n_units,) with ID of session in which each unit was recorded.
        - stimArray : array of shape (n_trials,) with stimulus values for each trial.
    '''
    if stim_type == 'AM':
        data0 = np.load(path + 'fr_arrays_AM.npz', allow_pickle=True)
    elif stim_type == 'PT':
        data0 = np.load(path + 'fr_arrays_pureTones.npz', allow_pickle=True)
    elif stim_type == 'NS':
        data0 = np.load(path + 'fr_arrays_naturalSound.npz', allow_pickle=True)  
    else:
        print('stim_type not valid.')

    # Repackage the data slightly into a new dictionary:
    data = {}
    for key in periods:
        data[key] = data0[key + 'fr'].T
        #print(data[key].shape)
    for key in ['brainRegionArray', 'mouseIDArray', 'sessionIDArray']:
        data[key] = data0[key]
    data['stimArray'] = data0['stimArray'][0,:]

    if pool_dorsal_posterior:
        pooled_name = 'Dor+Pos auditory area'
        region_array = data['brainRegionArray'].astype(object).copy()
        region_array[(region_array == 'Dorsal auditory area') | (region_array == 'Posterior auditory area')] = pooled_name
        data['brainRegionArray'] = region_array
    
    return data


def analyze_num_units(data, n_units_min=None, save_figs=False):
    '''
    Plot histograms showing the numbers of recorded units from each brain area.

    Parameters
    --
    data : A dictionary with the data.

    n_units_min : An integer denoting a threshold for how many units to keep from
        each brain area. It's only used for plotting, not for analysis.

    save_figs : If True, save figures as PDFs.
    '''
    sessionIDs = list(set(data['sessionIDArray']))
    regions_data = list(set(data['brainRegionArray']))
    regions = ['Primary auditory area',
               'Ventral auditory area',
               'Dorsal auditory area',
               'Posterior auditory area']
    if set(regions) != set(regions_data):
        print('Brain regions in data dictionary are different from expected.')

    n_units_dict = {}
    for region in regions:
        n_units_dict[region] = []
    for sessionID in sessionIDs:
        #print('\n')
        for region in regions:
            mask = np.logical_and(data['sessionIDArray']==sessionID, data['brainRegionArray']==region)
            n_units = data[periods[0]][:, mask].shape[1]
            #print((sessionID, region), ': ', n_units, 'units')
            n_units_dict[region].append(n_units)
    
    plt.figure(figsize=(5,8))
    for i, region in enumerate(regions):
        plt.subplot(int(411 + i))
        plt.hist(n_units_dict[region], bins=20)
        plt.title(region)
        plt.xlabel('Number of units per session')
        plt.ylabel('Number of sessions')
    plt.tight_layout()
    plt.show()
    if save_figs:
        plt.savefig('./figs/n_units_hist.pdf')
    
    n_sessions_dict = {}
    for region in regions:
        n_sessions_dict[region] = []
        for threshold in range(100):
            n_sessions_dict[region].append(np.sum(np.array(n_units_dict[region]) > threshold))
    
    plt.figure(figsize=(4,3))
    for region in regions:
        plt.plot(n_sessions_dict[region], label=region)
    plt.legend()
    plt.xlabel('Number of units')
    plt.ylabel('Number of sessions')
    plt.grid()
    if n_units_min is not None:
        plt.axvline(n_units_min, c='r', lw=2, ls=':')
    plt.tight_layout()
    if save_figs:
        plt.savefig('./figs/sessions_units.pdf')


def pca_plots(n_units_target, savefigs=False, use_umap=False, pool_dorsal_posterior=False):
    '''
    Make dimensionality reduction plots using PCA or UMAP, choosing a session for each brain region with approximately n_units_target units.
    '''

    ### Figure out which sessions have approximately the desired number of units.
    data = load_data('PT', pool_dorsal_posterior=pool_dorsal_posterior)
    sessionIDs = list(set(data['sessionIDArray']))
    n_units_dict = {}
    n_units_array = np.zeros((len(sessionIDs), len(regions)))
    for i, sessionID in enumerate(sessionIDs):
        for j, region in enumerate(regions):
            session_mask = data['sessionIDArray'] == sessionID
            region_mask = data['brainRegionArray'] == region
            n_units_dict[(sessionID, region)] = np.sum(np.logical_and(session_mask, region_mask))
            n_units_array[i, j] = np.sum(np.logical_and(session_mask, region_mask))
    #print(n_units_array)
    session_idxs = np.argmin((n_units_array - n_units_target)**2, axis=0)
    #print(session_idxs)
    #print([n_units_array[session_idxs[i], i] for i in range(len(regions))])
    pca_sessions = {}
    for i, region in enumerate(regions):
        pca_sessions[region] = sessionIDs[session_idxs[i]]
    #print(pca_sessions)

    dim_red_type = 'UMAP' if use_umap else 'PCA'
    if use_umap:
        xlabel = 'UMAP 1'
        ylabel = 'UMAP 2'
    else:
        xlabel = 'PC 1'
        ylabel = 'PC 2'

    ### Perform dimensionality reduction and plot results
    for stim_type in stim_types:
        data = load_data(stim_type, pool_dorsal_posterior=pool_dorsal_posterior)
        stimuli = np.sort(list(set(data['stimArray'])))
        for period in periods:
            plt.figure(figsize=(12,3.5))
            for j, region in enumerate(regions):
                plt.subplot(int(100 + 10*len(regions) + j + 1))
                plt.title(stim_type + ', ' + period + ', ' + region[:3])
                sessionID = pca_sessions[region]
                units_mask = data['sessionIDArray'] == sessionID
                spike_counts = data[period][:, units_mask]
                if use_umap:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
                    data_reduced = reducer.fit_transform(spike_counts)
                else:
                    pca = PCA(n_components=2)
                    data_reduced = pca.fit_transform(spike_counts)

                for i, stim in enumerate(stimuli):
                    plt.plot(data_reduced[:,0][data['stimArray']==stim], 
                            data_reduced[:,1][data['stimArray']==stim], '.',
                            color=(i / len(stimuli), 0, 1 - i / len(stimuli)),
                            label=str(round(stim, 1)) + ' Hz')
                    plt.xlabel(xlabel)
                    if j==0:
                        plt.ylabel(ylabel)
            if stim_type != 'NS':
                plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
            plt.tight_layout()
            if savefigs:
                plt.savefig('./figs/' + dim_red_type + '_' + stim_type + '_' + period + '.pdf')


def rsa(data, period, stim_type, n_units_min, n_samples=100, save_figs=False, pool_dorsal_posterior=False):    
    '''
    Performs representational similarity analysis on the data. Makes plots
    of representational dissimilarity matrices and a box plot of their off-
    diagonal values for each brain area.

    Parameters
    --
    data : A dictionary with the data.

    period : A string denoting the time window to analyze. Should be one of
        ['base', 'onset', 'sustained', 'offset'].

    stim_type : Type of stimulus. Should be one of ['AM', 'PT', 'NS'].

    n_units_min : Number of units to sample from each brain region in each session.

    n_samples : Number of times to randomly subsample populations.

    save_figs : If True, save figures as PDFs.

    pool_dorsal_posterior : If True, pool units from dorsal and posterior auditory areas.

    Returns
    --
    rdm_dict : A dictionary containing a representational dissimilarity matrix
        (a list of 2D arrays of shape n_stimuli-by-n_stimuli, with one such array per each session) 
        for each brain region. Each
        RDM is averaged over sessions (from all sessions in which at least n_units_min
        units were located in the brain region), as well as over repeated subsamplings
        of the n_units_min units in each brain region and session.

    rdm_flattened_dict : Same as rdm_dict, but where only above-the-diagonal entries
        from each RDM are kept and flattened into a 1D array.
    '''
    
    sessionIDs = list(set(data['sessionIDArray']))
    stimuli = np.sort(list(set(data['stimArray'])))

    regions_data = list(set(data['brainRegionArray']))

    if pool_dorsal_posterior or 'Dor+Pos auditory area' in regions_data:
        regions = ['Primary auditory area',
                'Ventral auditory area',
                'Dor+Pos auditory area']
    else:
        regions = ['Primary auditory area',
                'Ventral auditory area',
                'Dorsal auditory area',
                'Posterior auditory area']

    if set(regions) != set(regions_data):
        print('Brain regions in data dictionary are different from expected.')

    region_shortnames = [r[:3] if r != 'Dor+Pos auditory area' else 'Dor+Pos' for r in regions]

    spike_counts = data[period]
    spike_counts_zscored = (spike_counts - np.mean(spike_counts, axis=0)) / (1e-9 + np.std(spike_counts, axis=0))

    rdm_dict, rdm_flattened_dict = {}, {}
    for region in regions:
        #rdm_array = np.zeros((len(sessionIDs) * n_samples, len(stimuli), len(stimuli)))
        rdm_list = []
        for j, session in enumerate(sessionIDs):
            units_mask = np.logical_and(data['brainRegionArray'] == region, 
                                        data['sessionIDArray'] == session)
            n_units = spike_counts_zscored[:,units_mask].shape[1]
            active_units_mask = np.logical_and(units_mask, np.mean(data[period], axis=0) > 1e-9)
            n_units_active = np.sum(active_units_mask)
            rdm_array = np.zeros((n_samples,len(stimuli), len(stimuli)))
            if n_units_active > n_units_min-1:
                for k in range(n_samples):
                    x_bar = np.zeros((len(stimuli), n_units_min))
                    subsample_mask = [j < n_units_min for j in range(n_units_active)]
                    np.random.shuffle(subsample_mask)
                    for i, stimulus in enumerate(stimuli):
                        stim_mask = data['stimArray'] == stimulus
                        x = spike_counts_zscored[stim_mask, :][:, active_units_mask][:, subsample_mask]
                        #print(region, session, stimulus, x.shape)
                        x_bar[i, :] = np.mean(x, axis=0)  # average of trials of a given type
                    x_bar -= np.mean(x_bar, axis=0)  # mean-subtract the data before computing correlations
                    #rdm_array[j*n_samples+k, :, :] = 1 - np.corrcoef(x_bar)
                    rdm_array[k, :, :] = 1 - np.corrcoef(x_bar)
                rdm_list.append(rdm_array)
        #rdm_bar = np.mean(rdm_array, axis=0)  # average RDMs over sessions and subsamplings
        rdm_bar = [np.mean(rdm_array, axis=0) for rdm_array in rdm_list]  # average over subsamplings
        rdm_dict[region] = rdm_bar
        upper_diag = [np.triu(rdm, k=1) for rdm in rdm_bar]  # zero out elements on and below the diagonal
        #rdm_flattened_dict[region] = upper_diag[upper_diag!=0]  # remove zeros and flatten
        rdm_flattened_dict[region] = [ud[ud!=0] for ud in upper_diag]
    
    # Normalize all color bars to the same value:
    max_rdm_value = np.max([np.max(np.mean(np.array(rdm_dict[region]), axis=0)) for region in regions])
    
    plt.figure(figsize=(8,7))
    for i in range(len(regions)):
        plt.subplot(int(221 + i))
        plt.imshow(np.array(np.mean(np.array(rdm_dict[regions[i]]), axis=0)), vmin=0, vmax=max_rdm_value)
        plt.title(regions[i] + ', ' + period)
        plt.xticks(np.arange(len(stimuli))[::len(stimuli)-1], stimuli[::len(stimuli)-1]) # type: ignore
        plt.yticks(np.arange(len(stimuli))[::len(stimuli)-1], stimuli[::len(stimuli)-1])
        if stim_type == 'AM':
            plt.xlabel('AM freq. (Hz)')
            plt.ylabel('AM freq. (Hz)')
        elif stim_type == 'PT':
            plt.xlabel('PT freq. (Hz)')
            plt.ylabel('PT freq. (Hz)')
        elif stim_type == 'NS':
            plt.xlabel('Nat. sound ID')
            plt.ylabel('Nat. sound ID')
    
        plt.colorbar(label='Dissimilarity (1 - $\\rho$)', shrink=0.75)
    plt.tight_layout()
    if save_figs:
        plt.savefig('./figs/rdms_' + stim_type + '_' + period + '.pdf')
    
    plt.figure(figsize=(3,3))
    plt.ylabel('Dissimilarity (1 - $\\rho$)')
    y_vals = list(rdm_flattened_dict.values())
    y_vals_avg = []
    for i in range(len(regions)):
        y_vals_avg.append(np.mean(np.array(y_vals[i]), axis=0))
    for i in range(len(regions)):
        plt.plot((i+1) * np.ones(len(y_vals_avg[i])) + 0.05 * np.random.randn(len(y_vals_avg[i])), 
                 y_vals_avg[i], '.', c='k', alpha=0.1)
    #plt.boxplot(rdm_flattened_dict.values(), tick_labels=[key[:3] for key in rdm_flattened_dict.keys()]);
    plt.boxplot(y_vals_avg, tick_labels=region_shortnames);
    plt.title(stim_type + ', ' + period)
    plt.tight_layout()
    if save_figs:
        plt.savefig('./figs/rsa_boxplots_' + stim_type + '_' + period + '.pdf')

    # Compute significance with pairwise Mann-Whitney U test with Bonferroni corrections.
    #data_values = list(rdm_flattened_dict.values())
    data_values = y_vals_avg
    pairs = list(combinations(range(len(data_values)), 2))
    p_values = []
    print('\nStatistical comparisons for ' + stim_type + ', ' + period + '...')
    print(f"{'Comparison':<25} {'U-statistic':<12} {'p-value':<12} {'Significant'}")
    print("-" * 65)
    for i, j in pairs:
        u_stat, p_val = mannwhitneyu(data_values[i], data_values[j], 
                                    alternative='two-sided')
        p_values.append(p_val)
        
        # Bonferroni correction
        alpha = 0.05
        adjusted_alpha = alpha / len(pairs)
        is_sig = "Yes" if p_val < adjusted_alpha else "No"
        
        comparison = f"{region_shortnames[i]} vs {region_shortnames[j]}"
        print(f"{comparison:<25} {u_stat:<12.2f} {p_val:<12.4e} {is_sig}")
    print(f"\nBonferroni-corrected α = {adjusted_alpha:.4f}")

    return rdm_dict, rdm_flattened_dict


def stim_space_clustering(stim_type, region, period, sessionID, make_plots=False, pool_dorsal_posterior=False):
    """
    Perform clustering analysis in stimulus space for neural spike data.

    This function analyzes neural responses to different stimuli by averaging spike counts
    across trials for each stimulus type, z-scoring the responses, performing PCA,
    and evaluating clustering quality using silhouette scores. It compares the observed
    silhouette score against a null distribution generated from multivariate normal
    samples with the same covariance structure.

    Parameters:
    stim_type (str): Type of stimulus (e.g., 'NS', 'PT', 'AM').
    region (str): Brain region name (e.g., 'Primary auditory area').
    period (str): Time period for analysis (e.g., 'sustained', 'onset').
    sessionID (str): Session identifier.
    make_plots (bool, optional): If True, display plots and print intermediate results. Default is False.

    Returns:
    float: Z-scored silhouette score, indicating how many standard deviations the observed
           silhouette score is above the mean of the null distribution.
    """
    data = load_data(stim_type, pool_dorsal_posterior=pool_dorsal_posterior)
    
    session_mask = data['sessionIDArray'] == sessionID
    region_mask = data['brainRegionArray'] == region
    units_mask = np.logical_and(session_mask, region_mask)
    n_units = np.sum(units_mask)
    if make_plots:
        print(f'n_units = {n_units}')
    
    spike_counts = data[period][:, units_mask]
    z_responses = (spike_counts - np.mean(spike_counts, axis=0, keepdims=True)) / \
                     (np.std(spike_counts, axis=0, keepdims=True) + 1e-10)  # z score across trials for each unit   
    stims = data['stimArray']
    unique_stims = np.unique(stims) # Get unique stimulus values
    z_averaged = np.zeros((len(unique_stims), z_responses.shape[1]))
    for i, stim in enumerate(unique_stims):
        # Find trials with this stimulus
        mask = stims == stim
        # Average across those trials
        z_averaged[i, :] = np.mean(z_responses[mask, :], axis=0)
    z_responses = z_averaged.T  # Now shape is (n_units, n_stimuli)
    
    if make_plots:
        plt.figure(figsize=(5, 3))
        plt.plot(z_responses.T[:,::2])
        plt.xlabel('Stimulus')
        plt.ylabel('Response (z-scored)')
        plt.show()
    
    # Perform PCA on averaged_spike_counts
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(z_responses)
    
    if make_plots:
        # Plot PCA results
        plt.figure(figsize=(4, 4))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA of Averaged Spike Counts by Stimulus')
        plt.grid(True)
        plt.show()
        
        print(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
        print(f"Total variance explained by first 2 PCs: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Perform clustering
    sil_scores = []
    for n_clusters in range(2, 10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(z_responses)
        sil_scores.append(silhouette_score(z_responses, y_pred))
    best_sil_score = max(sil_scores)
    if make_plots:
        print(f"n_clusters = {sil_scores.index(best_sil_score) + 2}, Silhouette Score: {best_sil_score:.3f}")
        
        plt.figure(figsize=(5,3))
        plt.plot(range(2, 10), sil_scores, 'o-')
        plt.title('Silhouette Score vs Number of Clusters')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()
    
    # Compute null distribution of silhouette scores by sampling from a multivariate normal distribution
    n_iterations = 100
    null_sil_scores = []
    for _ in range(n_iterations):
        # Generate random data with same shape and covariance as z_responses
        random_data = np.random.multivariate_normal(
            mean=np.zeros(z_responses.shape[1]),
            cov=np.cov(z_responses, rowvar=False),
            size=z_responses.shape[0]
        )
        null_sil_scores_iter = []
        for n_clusters in range(2, 10):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            y_pred = kmeans.fit_predict(random_data)
            null_sil_scores_iter.append(silhouette_score(random_data, y_pred))
        best_sil_score_iter = max(null_sil_scores_iter)
        null_sil_scores.append(best_sil_score_iter)
    sil_score_z = (best_sil_score - np.mean(null_sil_scores)) / np.std(null_sil_scores)
    if make_plots:
        print(f'Silhouette score (z-scored): {sil_score_z}')
        
        plt.figure(figsize=(5,3))
        plt.hist(null_sil_scores, bins=30, alpha=0.7, label='Null Distribution')
        plt.axvline(best_sil_score, color='r', linestyle='--', label='Observed Silhouette Score')
        plt.xlabel('Silhouette Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    return best_sil_score, sil_score_z


def activity_space_clustering(region, period, sessionID, n_units_min, make_plots=False, pool_dorsal_posterior=False):
    """
    Compute silhouette scores for natural sounds using stimulus labels in activity space.
    
    This function analyzes clustering in activity space by using the actual stimulus
    labels rather than k-means clustering. It computes scores at two granularities:
    - Fine-grained: 20 stimulus labels (0-19)
    - Coarse-grained: 5 category labels (grouping 4 stimuli each)
    
    Performs 100 subsamples of n_units_min units and computes both raw and z-scored
    silhouette scores against null distributions.
    
    Parameters:
    region (str): Brain region name (e.g., 'Primary auditory area').
    period (str): Time period for analysis (e.g., 'sustained', 'onset').
    sessionID (str): Session identifier.
    n_units_min (int): Number of units to subsample.
    make_plots (bool, optional): If True, display plots. Default is False.
    
    Returns:
    dict: Dictionary with keys 'fine' and 'coarse', each containing [raw_score, z_score]
    """
    # Load natural sounds data
    data = load_data('NS', pool_dorsal_posterior=pool_dorsal_posterior)

    # Get session and region masks
    session_mask = data['sessionIDArray'] == sessionID
    region_mask = data['brainRegionArray'] == region
    units_mask = np.logical_and(session_mask, region_mask)
    n_units = np.sum(units_mask)
    
    if make_plots:
        print(f'n_units = {n_units}')
    
    # Get spike counts and stimulus labels
    spike_counts = data[period][:, units_mask]
    stims = data['stimArray']
    
    # Fine-grained labels: use stimulus IDs directly (0-19)
    fine_labels = stims.astype(int)
    
    # Coarse-grained labels: group every 4 stimuli into one category
    # 0-3 -> 0, 4-7 -> 1, 8-11 -> 2, 12-15 -> 3, 16-19 -> 4
    coarse_labels = (stims / 4).astype(int)
    
    # Subsample 100 times and compute scores
    n_iterations = 100
    fine_scores = []
    coarse_scores = []
    fine_z_scores = []
    coarse_z_scores = []
    
    # Number of null samples per subsample
    n_null_samples = 100
    
    for iteration in range(n_iterations):
        # Randomly select n_units_min units
        subsample_mask = np.random.choice(n_units, size=n_units_min, replace=False)
        spike_counts_sub = spike_counts[:, subsample_mask]
        
        # Z-score across trials for each unit
        z_responses = (spike_counts_sub - np.mean(spike_counts_sub, axis=0, keepdims=True)) / \
                         (np.std(spike_counts_sub, axis=0, keepdims=True) + 1e-10)
        
        # Compute silhouette scores for this subsample using actual stimulus labels
        fine_score = silhouette_score(z_responses, fine_labels)
        coarse_score = silhouette_score(z_responses, coarse_labels)
        fine_scores.append(fine_score)
        coarse_scores.append(coarse_score)
        
        # Compute null distribution for this subsample
        null_fine_scores = []
        null_coarse_scores = []
        cov_matrix = np.cov(z_responses, rowvar=False)
        
        for _ in range(n_null_samples):
            # Generate random data with same shape and covariance as this subsample
            try:
                random_data = np.random.multivariate_normal(
                    mean=np.zeros(z_responses.shape[1]),
                    cov=cov_matrix,
                    size=z_responses.shape[0]
                )
                
                # Use k-means to cluster the random data
                # Fine-grained: k=20 clusters
                kmeans_fine = KMeans(n_clusters=20, random_state=42, n_init=10)
                labels_fine = kmeans_fine.fit_predict(random_data)
                null_fine_scores.append(silhouette_score(random_data, labels_fine))
                
                # Coarse-grained: k=5 clusters
                kmeans_coarse = KMeans(n_clusters=5, random_state=42, n_init=10)
                labels_coarse = kmeans_coarse.fit_predict(random_data)
                null_coarse_scores.append(silhouette_score(random_data, labels_coarse))
                
            except (np.linalg.LinAlgError, ValueError):
                # If covariance matrix is not positive definite, skip this sample
                continue
        
        # Compute z-scores for this subsample
        if len(null_fine_scores) > 0:
            fine_z = (fine_score - np.mean(null_fine_scores)) / (np.std(null_fine_scores) + 1e-10)
            coarse_z = (coarse_score - np.mean(null_coarse_scores)) / (np.std(null_coarse_scores) + 1e-10)
            fine_z_scores.append(fine_z)
            coarse_z_scores.append(coarse_z)
    
    # Average the scores
    fine_score_avg = np.mean(fine_scores)
    coarse_score_avg = np.mean(coarse_scores)
    fine_z_score_avg = np.mean(fine_z_scores)
    coarse_z_score_avg = np.mean(coarse_z_scores)
    
    if make_plots:
        print(f'Fine-grained: raw={fine_score_avg:.3f}, z={fine_z_score_avg:.3f}')
        print(f'Coarse-grained: raw={coarse_score_avg:.3f}, z={coarse_z_score_avg:.3f}')
        
        # Plot PCA colored by fine and coarse labels using one subsample
        from sklearn.decomposition import PCA
        subsample_mask = np.random.choice(n_units, size=n_units_min, replace=False)
        spike_counts_sub = spike_counts[:, subsample_mask]
        z_responses = (spike_counts_sub - np.mean(spike_counts_sub, axis=0, keepdims=True)) / \
                         (np.std(spike_counts_sub, axis=0, keepdims=True) + 1e-10)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(z_responses)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Fine-grained (20 colors)
        scatter1 = axes[0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=fine_labels, cmap='tab20', alpha=0.6, s=30)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0].set_title(f'Fine-grained (20 stimuli)\nraw={fine_score_avg:.3f}, z={fine_z_score_avg:.3f}')
        plt.colorbar(scatter1, ax=axes[0], label='Stimulus ID')
        
        # Coarse-grained (5 colors)
        scatter2 = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                   c=coarse_labels, cmap='Set1', vmin=0, vmax=4, alpha=0.6, s=30)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1].set_title(f'Coarse-grained (5 categories)\nraw={coarse_score_avg:.3f}, z={coarse_z_score_avg:.3f}')
        cbar = plt.colorbar(scatter2, ax=axes[1], label='Category ID', ticks=[0, 1, 2, 3, 4])
        
        plt.tight_layout()
        plt.show()
        
        # Plot distributions of raw and z-scored values across subsamples
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].hist(fine_scores, bins=30, alpha=0.7)
        axes[0, 0].axvline(fine_score_avg, color='r', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].set_xlabel('Silhouette Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Fine-grained Raw Scores (across subsamples)')
        axes[0, 0].legend()
        
        axes[0, 1].hist(fine_z_scores, bins=30, alpha=0.7)
        axes[0, 1].axvline(fine_z_score_avg, color='r', linestyle='--', linewidth=2, label='Mean')
        axes[0, 1].set_xlabel('Z-Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Fine-grained Z-Scores (across subsamples)')
        axes[0, 1].legend()
        
        axes[1, 0].hist(coarse_scores, bins=30, alpha=0.7)
        axes[1, 0].axvline(coarse_score_avg, color='r', linestyle='--', linewidth=2, label='Mean')
        axes[1, 0].set_xlabel('Silhouette Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Coarse-grained Raw Scores (across subsamples)')
        axes[1, 0].legend()
        
        axes[1, 1].hist(coarse_z_scores, bins=30, alpha=0.7)
        axes[1, 1].axvline(coarse_z_score_avg, color='r', linestyle='--', linewidth=2, label='Mean')
        axes[1, 1].set_xlabel('Z-Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Coarse-grained Z-Scores (across subsamples)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return {'fine': [fine_score_avg, fine_z_score_avg], 'coarse': [coarse_score_avg, coarse_z_score_avg]}



def _compute_session_spearmans_from_dissim(dissim: dict):
    """Compute per-session Spearman correlations for each (stim_type, period) and region-pair.

    Returns dict keyed by (stim_type, period) -> {(regionA, regionB): np.ndarray_of_rhos_per_session}
    """
    import collections
    from scipy import stats
    import numpy as np

    keys = list(dissim.keys())
    if not keys:
        return {}
    regions_by_sp = collections.defaultdict(set)
    stim_periods = collections.defaultdict(set)
    for (stim_type, period, region) in keys:
        stim_periods[stim_type].add(period)
        regions_by_sp[(stim_type, period)].add(region)

    out = {}
    for stim_type, periods in stim_periods.items():
        for period in periods:
            sp = (stim_type, period)
            regions = sorted(regions_by_sp[sp])
            if not regions:
                continue
            first_key = (stim_type, period, regions[0])
            sessions = len(dissim[first_key])
            pair_rhos = {}
            for i in range(len(regions)):
                for j in range(i + 1, len(regions)):
                    ra, rb = regions[i], regions[j]
                    rhos = []
                    for s in range(sessions):
                        a = dissim.get((stim_type, period, ra), None)
                        b = dissim.get((stim_type, period, rb), None)
                        if a is None or b is None:
                            rhos.append(np.nan)
                            continue
                        try:
                            va = np.asarray(a[s])
                            vb = np.asarray(b[s])
                        except Exception:
                            rhos.append(np.nan)
                            continue
                        if va.size != vb.size or va.size < 2:
                            rhos.append(np.nan)
                            continue
                        mask = np.isfinite(va) & np.isfinite(vb)
                        if mask.sum() < 2:
                            rhos.append(np.nan)
                            continue
                        try:
                            rho = stats.spearmanr(va[mask], vb[mask]).correlation
                        except Exception:
                            rho = np.nan
                        rhos.append(float(rho) if rho is not None else np.nan)
                    pair_rhos[(ra, rb)] = np.asarray(rhos, dtype=float)
            out[sp] = pair_rhos
    return out


def pairwise_region_comparisons(dissim_dict: dict, results_type: str = 'dissim'):
    """Compute Spearman per-session correlations for region pairs and run Mann-Whitney U tests.

    Returns (sig_comparisons, all_results) where:
      - sig_comparisons: dict keyed by (stim_type, period) -> list of significant pair tuples
      - all_results: dict keyed by (stim_type, period) -> list of result dicts for each pair
    """
    import numpy as np
    from scipy.stats import mannwhitneyu

    sp_pair_rhos = _compute_session_spearmans_from_dissim(dissim_dict)
    all_results = {}
    sig_comparisons = {}
    for sp, pair_rhos in sp_pair_rhos.items():
        pairs = list(pair_rhos.keys())
        n_pairs = len(pairs)
        results_list = []
        cleaned = {p: pair_rhos[p][np.isfinite(pair_rhos[p])] for p in pairs}
        for p in pairs:
            x = cleaned[p]
            others = np.concatenate([cleaned[q] for q in pairs if q != p and cleaned[q].size > 0]) if n_pairs > 1 else np.array([])
            res = {
                'region1': p[0],
                'region2': p[1],
                'n': int(x.size),
                'mean': float(np.nan) if x.size == 0 else float(np.mean(x)),
                'median': float(np.nan) if x.size == 0 else float(np.median(x)),
                'rhos': x.tolist(),
                'p_value': np.nan,
                'p_bonf': np.nan,
                'significant': False,
            }
            if x.size > 0 and others.size > 0:
                try:
                    stat, pval = mannwhitneyu(x, others, alternative='two-sided')
                except Exception:
                    pval = np.nan
                p_bonf = min(1.0, float(pval) * n_pairs) if np.isfinite(pval) else np.nan
                res['p_value'] = float(pval) if np.isfinite(pval) else np.nan
                res['p_bonf'] = p_bonf
                res['significant'] = np.isfinite(p_bonf) and (p_bonf < 0.05)
            results_list.append(res)
        all_results[sp] = results_list
        sig_comparisons[sp] = [r for r in results_list if r['significant']]
    return sig_comparisons, all_results


def print_pairwise_results(sig_comparisons: dict, results_type: str = 'dissim'):
    """Print significant comparisons in a compact form."""
    if not sig_comparisons:
        print('No comparisons to print.')
        return
    print(f"Significant region-pair comparisons ({results_type}) (Bonferroni-corrected p < 0.05):")
    for sp, sigs in sig_comparisons.items():
        if not sigs:
            continue
        stim_type, period = sp
        for r in sigs:
            print(f"  stim={stim_type}, period={period}: {r['region1']} vs {r['region2']} -> p_bonf={r['p_bonf']:.4g}, n={r['n']}, mean_rho={r['mean']:.4g}")


def save_pairwise_results(all_results: dict, results_type: str = 'dissim', outdir: str = 'results'):
    """Save all_results to `outdir/region_pair_correlations_<timestamp>.pkl`."""
    import os
    import pickle
    import datetime

    os.makedirs(outdir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = os.path.join(outdir, f'region_pair_correlations_{results_type}_{ts}.pkl')
    with open(outpath, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'Saved full results to {outpath}')


def classification_accuracy(X, y, n_splits=5, random_state=42):
    """
    Train a multi-class logistic regression classifier with cross-validated hyperparameter selection.
    
    Parameters
    ----------
    X : array-like, shape (n_trials, n_features)
        Feature matrix where each row is a trial and each column is a feature (e.g., neuron firing rate)
    y : array-like, shape (n_trials,)
        Labels for each trial. Each unique value is treated as a separate class.
    n_splits : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    accuracy : float
        Classification accuracy (mean accuracy on held-out data across CV folds)
    """
    import numpy as np
    import warnings
    
    # Suppress specific sklearn warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
    
    # Convert continuous labels to discrete classes if needed
    # Treat each unique value as a separate class
    unique_labels = np.unique(y)
    label_to_class = {label: idx for idx, label in enumerate(unique_labels)}
    y_discrete = np.array([label_to_class[label] for label in y])
    
    # Create pipeline with scaling and logistic regression
    # Use multinomial loss which is the recommended approach for multiclass problems
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            penalty='l2',
            max_iter=5000,
            multi_class='multinomial',
            solver='lbfgs',  # Set default solver here
            random_state=random_state
        ))
    ])
    
    # Define hyperparameter grid for L2 regularization strength
    # Only use solvers that support multinomial: lbfgs, newton-cg, sag, saga
    param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'classifier__solver': ['lbfgs', 'newton-cg']  # Both support multinomial and are stable
    }
    
    # Use stratified k-fold to maintain class proportions
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Evaluate using nested cross-validation on the full dataset
    # This gives us an unbiased estimate of generalization performance
    scores = cross_val_score(grid_search, X, y_discrete, cv=cv, scoring='accuracy')
    
    # Return mean accuracy across folds
    return np.mean(scores)


def ridge_log_stim_r2(X, y, n_splits=5, random_state=42, alphas=None):
    """
    Decode log stimulus value using linear ridge regression with nested CV.
    Returns mean r^2 across outer folds, where r is Pearson correlation between
    predicted and true y on each outer fold.
    """
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from scipy.stats import pearsonr

    y = np.asarray(y, dtype=float)
    y = np.log(y)

    if alphas is None:
        alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge())
    ])

    param_grid = {'ridge__alpha': alphas}
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='neg_mean_squared_error'
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        if np.std(y_test) < 1e-12 or np.std(y_pred) < 1e-12:
            r2_scores.append(np.nan)
            continue

        r = pearsonr(y_test, y_pred).statistic
        r2_scores.append(float(r**2))

    r2_scores = np.asarray(r2_scores, dtype=float)
    return float(np.nanmean(r2_scores))


def kernel_ridge_log_stim_r2(X, y, n_splits=5, random_state=42, alphas=None, gammas=None):
    """
    Decode log stimulus value using kernel ridge regression (RBF) with nested CV.
    Returns mean r^2 across outer folds, where r is Pearson correlation between
    predicted and true y on each outer fold.
    """
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from scipy.stats import pearsonr

    y = np.asarray(y, dtype=float)
    y = np.log(y)

    if alphas is None:
        alphas = [1e-3, 1e-2, 1e-1, 1, 10, 100]

    if gammas is None:
        base_gamma = 1.0 / max(1, X.shape[1])  # scale heuristic with standardized features
        gammas = [base_gamma * g for g in [0.1, 0.3, 1.0, 3.0, 10.0]]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('krr', KernelRidge(kernel='rbf'))
    ])

    param_grid = {'krr__alpha': alphas, 'krr__gamma': gammas}
    inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2_scores = []
    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='neg_mean_squared_error'
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        if np.std(y_test) < 1e-12 or np.std(y_pred) < 1e-12:
            r2_scores.append(np.nan)
            continue

        r = pearsonr(y_test, y_pred).statistic
        r2_scores.append(float(r**2))

    r2_scores = np.asarray(r2_scores, dtype=float)
    return float(np.nanmean(r2_scores))
   

def ns_coarse_classification_accuracy(X, y):
    y_int = np.asarray(y).astype(int)
    y_coarse = y_int // 4  # 5 categories
    return classification_accuracy(X, y_coarse)


def ns_fine_classification_accuracy(X, y):
    y_int = np.asarray(y).astype(int)
    accuracies = []
    for cat in range(5):
        mask = (y_int // 4) == cat
        if np.sum(mask) < 2:
            continue
        y_sub = y_int[mask] - 4 * cat  # 0..3 within category
        if len(np.unique(y_sub)) < 2:
            continue
        X_sub = X[mask]
        accuracies.append(classification_accuracy(X_sub, y_sub))
    return float(np.mean(accuracies)) if accuracies else np.nan


def stim_space_clustering_from_data(spike_counts, stim_array, make_plots=False,
                                    random_state=42, n_null=100, k_range=range(2, 10)):
    """
    Compute raw and z-scored silhouette score from spike_counts and stim_array.
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # z-score across trials per unit
    z_responses = (spike_counts - np.mean(spike_counts, axis=0, keepdims=True)) / \
                  (np.std(spike_counts, axis=0, keepdims=True) + 1e-10)

    unique_stims = np.unique(stim_array)
    z_averaged = np.zeros((len(unique_stims), z_responses.shape[1]))
    for i, stim in enumerate(unique_stims):
        mask = stim_array == stim
        z_averaged[i, :] = np.mean(z_responses[mask, :], axis=0)

    # Keep behavior consistent with existing function
    z_responses = z_averaged.T  # (n_units, n_stimuli)

    if make_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 3))
        plt.plot(z_responses.T[:, ::2])
        plt.xlabel('Stimulus')
        plt.ylabel('Response (z-scored)')
        plt.show()

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(z_responses)
        plt.figure(figsize=(4, 4))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA of Averaged Spike Counts by Stimulus')
        plt.grid(True)
        plt.show()

    sil_scores = []
    for n_clusters in k_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        y_pred = kmeans.fit_predict(z_responses)
        sil_scores.append(silhouette_score(z_responses, y_pred))
    best_sil_score = max(sil_scores)

    rng = np.random.default_rng(random_state)
    null_sil_scores = []
    for _ in range(n_null):
        random_data = rng.multivariate_normal(
            mean=np.zeros(z_responses.shape[1]),
            cov=np.cov(z_responses, rowvar=False),
            size=z_responses.shape[0]
        )
        null_scores_iter = []
        for n_clusters in k_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            y_pred = kmeans.fit_predict(random_data)
            null_scores_iter.append(silhouette_score(random_data, y_pred))
        null_sil_scores.append(max(null_scores_iter))

    sil_score_z = (best_sil_score - np.mean(null_sil_scores)) / np.std(null_sil_scores)
    return best_sil_score, sil_score_z


def participation_ratio(X):
    import numpy as np
    C = np.cov(X.T)
    eigenvals = np.linalg.eigvals(C)
    if np.linalg.norm(np.imag(eigenvals)) < 1e-8:
        eigenvals = np.real(eigenvals)
    return (np.sum(eigenvals))**2 / np.sum(eigenvals**2)


def _aggregate_metric_values(values):
    if not values:
        return np.nan
    first = values[0]
    if np.isscalar(first):
        return float(np.mean(values))
    arr = np.asarray(values, dtype=float)
    return arr.mean(axis=0).tolist()


def compute_metric_results(metric_fn, n_units_min, n_subsamples=1, stim_types=None,
                           periods=None, regions=None, random_state=42, include_y=False,
                           pool_dorsal_posterior=False, verbose=True):
    """
    Compute metric results across sessions, brain regions, stimulus types, and time periods.

    This function processes neural data by subsampling units and applying a metric function
    to compute results for different experimental conditions.

    Parameters
    ----------
    metric_fn : callable
        A function that computes a metric given input data. Should accept parameters:
        X (array-like), y (optional array-like), and keyword arguments stim_type, period,
        region, and sessionID.
    n_units_min : int
        Minimum number of units required for analysis. Units will be subsampled to this size.
    n_subsamples : int, optional
        Number of times to subsample units for each condition. Default is 1.
    stim_types : list of str, optional
        List of stimulus types to analyze. If None, uses global 'stim_types' variable.
    periods : list of str, optional
        List of time periods to analyze. If None, uses global 'periods' variable.
    regions : list of str, optional
        List of brain regions to analyze. If None, uses global 'regions' variable.
    random_state : int, optional
        Seed for random number generator to ensure reproducibility. Default is 42.
    include_y : bool, optional
        Whether to include stimulus labels (y) when calling metric_fn. Default is False.
    verbose : bool, optional
        Whether to print progress messages during processing. Default is True.

    Returns
    -------
    dict
        A dictionary with keys (period, stim_type, region) and values as lists of
        aggregated metric results across sessions. Each element corresponds to one
        session's result after aggregating n_subsamples.

    Notes
    -----
    - Requires a load_data() function to be available in the scope.
    - Requires an _aggregate_metric_values() function to combine subsample results.
    - Only processes sessions/regions with at least n_units_min units available.
    - Units are randomly subsampled without replacement for each subsample iteration.
    """
    import numpy as np

    if stim_types is None:
        stim_types = globals().get('stim_types', [])
    if periods is None:
        periods = globals().get('periods', [])
    if regions is None:
        regions = globals().get('regions', [])

    results = {(period, stim_type, region): [] for stim_type in stim_types for period in periods for region in regions}
    rng = np.random.default_rng(random_state)

    sessionIDs = list(set(load_data('PT', pool_dorsal_posterior=pool_dorsal_posterior)['sessionIDArray']))
    for session in sessionIDs:
        if verbose:
            print(f'Processing session {session}...')
        for stim_type in stim_types:
            data = load_data(stim_type, pool_dorsal_posterior=pool_dorsal_posterior)            
            for region in regions:
                session_mask = data['sessionIDArray'] == session
                region_mask = data['brainRegionArray'] == region
                units_mask = np.logical_and(session_mask, region_mask)
                n_units = np.sum(units_mask)
                for period in periods:
                    active_units_mask = np.logical_and(units_mask, np.mean(data[period], axis=0) > 1e-9)
                    n_units_active = np.sum(active_units_mask)
                    if n_units_active >= n_units_min:
                        metric_vals = []
                        for _ in range(n_subsamples):
                            selected_units = rng.choice(np.where(active_units_mask)[0], size=n_units_min, replace=False)
                            X = data[period][:, selected_units]
                            y = data['stimArray'] if include_y else None
                            metric_vals.append(metric_fn(X, y, stim_type=stim_type, period=period,
                                                         region=region, sessionID=session))
                        results[(period, stim_type, region)].append(_aggregate_metric_values(metric_vals))
    return results


def _extract_metric_list(values, value_index=None):
    if value_index is None:
        return values
    return [v[value_index] for v in values] if values else []


def grouped_positions(n_groups, group_size, gap=1, start=1):
    positions = []
    pos = start
    for _ in range(n_groups):
        for _ in range(group_size):
            positions.append(pos)
            pos += 1
        pos += gap
    return positions


def collect_boxplot_data(results_dict, period, stim_types, regions, value_index=None, region_label_map=None):
    data, labels = [], []
    for stim_type in stim_types:
        for region in regions:
            key = (period, stim_type, region)
            if key in results_dict and results_dict[key]:
                vals = _extract_metric_list(results_dict[key], value_index=value_index)
                data.append(vals)
                if region_label_map:
                    labels.append(f"{stim_type} - {region_label_map.get(region, region[:3])}")
                else:
                    labels.append(f"{stim_type} - {region[:3]}")
    return data, labels


def plot_metric_boxplot(results_dict, period, stim_types, regions, value_index=None,
                        ax=None, title=None, ylabel=None, colors=None, region_label_map=None,
                        chance_level=None):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    data, labels = collect_boxplot_data(
        results_dict, period, stim_types, regions,
        value_index=value_index,
        region_label_map=region_label_map
    )
    positions = grouped_positions(len(stim_types), len(regions), gap=1)

    if data:
        boxes = ax.boxplot(data, positions=positions, patch_artist=True,
                           medianprops=dict(color='black'))
        for i, box in enumerate(boxes['boxes']):
            box.set_facecolor(colors[i % len(regions)])
    if chance_level is not None:
        ax.axhline(chance_level, linestyle='--', color='black', linewidth=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

def pairwise_region_comparisons_scalar(results_dict, stim_types=None, periods=None, regions=None,
                                       value_index=None):
    import numpy as np
    from scipy.stats import mannwhitneyu
    from itertools import combinations

    if stim_types is None:
        stim_types = globals().get('stim_types', [])
    if periods is None:
        periods = globals().get('periods', [])
    if regions is None:
        regions = globals().get('regions', [])

    sig_comparisons = {}
    all_results = {}

    for stim_type in stim_types:
        for period in periods:
            key_prefix = (period, stim_type)
            all_results[key_prefix] = []
            sig_comparisons[key_prefix] = []

            region_data = {}
            for region in regions:
                key = (period, stim_type, region)
                if key in results_dict and results_dict[key]:
                    region_data[region] = np.asarray(_extract_metric_list(results_dict[key], value_index), dtype=float)

            if len(region_data) < 2:
                continue

            region_pairs = list(combinations(region_data.keys(), 2))
            n_comparisons = len(region_pairs)
            for region1, region2 in region_pairs:
                data1 = region_data[region1]
                data2 = region_data[region2]
                stat, pval = mannwhitneyu(data1, data2, alternative='two-sided')

                result = {
                    'period': period,
                    'stim_type': stim_type,
                    'region1': region1,
                    'region2': region2,
                    'n1': len(data1),
                    'n2': len(data2),
                    'median1': float(np.median(data1)) if len(data1) else np.nan,
                    'median2': float(np.median(data2)) if len(data2) else np.nan,
                    'mean1': float(np.mean(data1)) if len(data1) else np.nan,
                    'mean2': float(np.mean(data2)) if len(data2) else np.nan,
                    'U_stat': float(stat),
                    'p_value': float(pval),
                    'p_bonf': min(1.0, float(pval) * n_comparisons),
                    'significant': float(pval) < (0.05 / n_comparisons)
                }

                all_results[key_prefix].append(result)
                if result['significant']:
                    sig_comparisons[key_prefix].append(result)

    return sig_comparisons, all_results


def print_pairwise_results_scalar(sig_comparisons, results_type='metric'):
    if not sig_comparisons:
        print('No comparisons to print.')
        return

    print(f"\nSignificant pairwise regional comparisons ({results_type})")
    print("=" * 70)
    for (period, stim_type), results in sig_comparisons.items():
        if not results:
            continue
        print(f"{stim_type} - {period.upper()}")
        print("-" * 70)
        for r in results:
            print(f"  {r['region1'][:15]:15} vs {r['region2'][:15]:15} "
                  f"p={r['p_value']:.4e} (bonf={r['p_bonf']:.4e})")