import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

# Create save directory
save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters/anova'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Directory containing HDF5 files
data_directory = "/Users/emmabelhadfa/Documents/Oxford/orex/OTES/data/locations/"

# Get all HDF5 file paths in the directorypi
file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".hdf5")]

# Define band parameter regions
bp1_region = (950, 1150)  # BP1 region
bp2_region = (800, 950)   # BP2 region 
bp3_region = (1150, 1250) # BP3 region (for slope calculation)
bp4_region = (400, 500)   # BP4 region

# Dictionary to store band parameters for each site
band_params_by_site = {}

# Process each file
for file_path in file_paths:
    site_name = os.path.basename(file_path).split('.')[0]
    
    with h5py.File(file_path, "r") as file:
        mt_emissivity = file["mt_emissivity"][:, :, :]
        xaxis_L3 = file["xaxis_L3"][:, 0, 0]

    # Clean and process data
    mt_emissivity_clipped = np.clip(mt_emissivity, -1.2, 1.2)
    means = np.mean(mt_emissivity_clipped, axis=(0, 1))
    z_scores = (means - np.mean(means)) / np.std(means)
    non_outliers = np.where(np.abs(z_scores) <= 3)[0]
    # Filter non_outliers to be within bounds of both arrays
    max_idx = min(mt_emissivity_clipped.shape[2], len(xaxis_L3)) - 1
    non_outliers = non_outliers[non_outliers <= max_idx]
    non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers]
    
    # Initialize data_list if it doesn't exist
    if 'data_list' not in locals():
        data_list = []

    # Instead of taking the site average, keep individual measurements
    for i in range(non_outlier_emissivity.shape[0]):
        spectrum = non_outlier_emissivity[i, 0, :]
        
        # Calculate band parameters for this spectrum
        bp_values = []
        for region in [bp1_region, bp2_region, bp4_region]:
            mask = (xaxis_L3[non_outliers] >= region[0]) & (xaxis_L3[non_outliers] <= region[1])
            bp_value = np.mean(spectrum[mask])
            bp_values.append(bp_value)
        
        # Calculate slope (BP3)
        slope_mask = (xaxis_L3[non_outliers] >= bp3_region[0]) & (xaxis_L3[non_outliers] <= bp3_region[1])
        slope_x = xaxis_L3[non_outliers][slope_mask]
        slope_y = spectrum[slope_mask]
        slope, _ = np.polyfit(slope_x, slope_y, 1)
        bp_values.insert(2, slope)
        
        # Add to data list
        data_list.append({
            'Site': site_name,
            'BP1': bp_values[0],
            'BP2': bp_values[1],
            'BP3': bp_values[2],
            'BP4': bp_values[3]
        })

# Convert data to pandas DataFrame for easier analysis
df = pd.DataFrame(data_list)

# Print diagnostic information before ANOVA
print("\nDiagnostic Information:")
print("Number of sites:", len(df['Site'].unique()))
print("\nData shape:", df.shape)
print("\nSample of data:")
print(df.head())
print("\nChecking for NaN values:")
print(df.isna().sum())

# Perform one-way ANOVA for each band parameter
bp_names = ['BP1', 'BP2', 'BP3', 'BP4']
anova_results = {}

for bp in bp_names:
    groups = [group[bp].values for name, group in df.groupby('Site')]
    # Print diagnostic information for each group
    print(f"\n{bp} groups:")
    for i, g in enumerate(groups):
        print(f"Group {i} size: {len(g)}, mean: {np.mean(g)}, contains NaN: {np.isnan(g).any()}")
    
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results[bp] = {'F-statistic': f_stat, 'p-value': p_val}

# Create visualization
plt.figure(figsize=(12, 6))

# Create box plot
plt.subplot(121)
df_melted = pd.melt(df, id_vars=['Site'], value_vars=['BP1', 'BP2', 'BP3', 'BP4'])
sns.boxplot(x='variable', y='value', hue='Site', data=df_melted)
plt.title('Band Parameters Distribution by Site')
plt.xlabel('Band Parameter')
plt.ylabel('Value')

# Create heatmap of F-statistics
plt.subplot(122)
f_stats = np.array([[anova_results[bp]['F-statistic']] for bp in bp_names])
sns.heatmap(f_stats, 
            annot=True, 
            fmt='.4f',
            cmap='YlOrRd',
            yticklabels=bp_names,
            xticklabels=['F-statistic'])
plt.title('ANOVA F-statistics by Band Parameter')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'anova_summary.png'))
plt.show()

# Print ANOVA results and perform Tukey's test
print("\nOne-way ANOVA and Tukey HSD Results:")
for bp in bp_names:
    print(f"\n{bp}:")
    print(f"F-statistic: {anova_results[bp]['F-statistic']:.4f}")
    print(f"p-value: {anova_results[bp]['p-value']:.4f}")
    
    # Perform Tukey's test
    tukey = pairwise_tukeyhsd(df[bp], df['Site'])
    print("\nTukey HSD pairwise comparisons:")
    print(tukey)

# Create an additional visualization for pairwise comparisons
plt.figure(figsize=(15, 10))
for i, bp in enumerate(bp_names, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Site', y=bp, data=df)
    plt.title(f'{bp} by Site')
    plt.xticks(rotation=45)
    plt.tight_layout()

plt.show()

# Calculate pairwise distances between sites
sites = df['Site'].unique()
n_sites = len(sites)
distance_matrix = np.zeros((n_sites, n_sites))

for i, site1 in enumerate(sites):
    for j, site2 in enumerate(sites[i+1:], i+1):
        site1_data = df[df['Site'] == site1][bp_names].values
        site2_data = df[df['Site'] == site2][bp_names].values
        
        # Calculate Euclidean distance between site centroids
        centroid_dist = np.linalg.norm(np.mean(site1_data, axis=0) - np.mean(site2_data, axis=0))
        distance_matrix[i, j] = centroid_dist
        distance_matrix[j, i] = centroid_dist

# Save results to text file
with open(os.path.join(save_dir, 'anova_analysis_results.txt'), 'w') as f:
    f.write("ANOVA Analysis Results\n")
    f.write("=====================\n\n")
    
    f.write("Dataset Summary\n")
    f.write("--------------\n")
    f.write(f"Total number of spectra: {len(df)}\n")
    f.write("Number of spectra per site:\n")
    for site in sites:
        f.write(f"{site}: {len(df[df['Site'] == site])}\n")
    
    # Add statistical analysis results
    f.write("\nStatistical Analysis\n")
    f.write("-------------------\n")
    for site in sites:
        site_data = df[df['Site'] == site]
        f.write(f"\n{site} Statistics:\n")
        for bp in bp_names:
            mean = site_data[bp].mean()
            std = site_data[bp].std()
            f.write(f"{bp:>8}: mean = {mean:.3f}, std = {std:.3f}\n")
    
    # Add ANOVA results with statistical significance
    f.write("\nOne-way ANOVA Results\n")
    f.write("--------------------\n")
    for bp in bp_names:
        f.write(f"\n{bp}:\n")
        f.write(f"F-statistic: {anova_results[bp]['F-statistic']:.4f}\n")
        f.write(f"p-value: {anova_results[bp]['p-value']:.4e}\n")
        
        # Add interpretation of significance
        p_val = anova_results[bp]['p-value']
        if p_val < 0.001:
            significance = "Highly significant"
        elif p_val < 0.01:
            significance = "Very significant"
        elif p_val < 0.05:
            significance = "Significant"
        else:
            significance = "Not significant"
        f.write(f"Significance: {significance} (α = 0.05)\n")
    
    # Add Tukey's test results with significance levels
    f.write("\nTukey HSD Results\n")
    f.write("----------------\n")
    for bp in bp_names:
        f.write(f"\n{bp} Pairwise Comparisons:\n")
        tukey = pairwise_tukeyhsd(df[bp], df['Site'])
        
        # Extract and format Tukey results
        for comparison in tukey._results_table.data[1:]:  # Skip header
            # Get only the needed values from the comparison
            group1, group2, meandiff, lower, upper, reject = comparison[:6]  # Limit to first 6 values
            p_value = tukey.pvalues[tukey._results_table.data[1:].index(comparison)]
            
            f.write(f"\n{group1} vs {group2}:\n")
            f.write(f"  Mean difference: {meandiff:.4f}\n")
            f.write(f"  95% CI: [{lower:.4f}, {upper:.4f}]\n")
            f.write(f"  p-value: {p_value:.4e}\n")
            f.write(f"  Significant: {'Yes' if reject else 'No'} (α = 0.05)\n")
    
    # Add pairwise distances
    f.write("\nPairwise Distances Between Sites\n")
    f.write("-----------------------------\n")
    for i, site1 in enumerate(sites):
        for j, site2 in enumerate(sites[i+1:], i+1):
            f.write(f"{site1} vs {site2}: {distance_matrix[i,j]:.3f}\n")

# Create heatmap of site distances
plt.figure(figsize=(10, 8))
plt.imshow(distance_matrix, cmap='viridis')
plt.colorbar(label='Euclidean Distance')
plt.xticks(range(n_sites), sites, rotation=45)
plt.yticks(range(n_sites), sites)
plt.title('Pairwise Site Distances')

for i in range(n_sites):
    for j in range(n_sites):
        plt.text(j, i, f'{distance_matrix[i,j]:.4f}',
                ha='center', va='center',
                color='white' if distance_matrix[i,j] > np.mean(distance_matrix) else 'black')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'site_distances_heatmap.png'))
plt.show()

# Create pairwise plots
n_features = len(bp_names)
fig = plt.figure(figsize=(12, 12))
plt.subplots_adjust(top=0.82)  # Increased space for title even more
plot_idx = 1

for i in range(n_features):
    for j in range(i+1, n_features):
        plt.subplot(n_features-1, n_features-1, plot_idx)
        
        sns.scatterplot(data=df, x=bp_names[i], y=bp_names[j], 
                       hue='Site', alpha=0.5)
        
        plt.grid(True, linestyle='--', alpha=0.4)
        if plot_idx > (n_features-2)*(n_features-1):  # Only show legend for bottom row
            plt.legend(title="Sites", bbox_to_anchor=(1.05, 0))
        else:
            plt.legend([],[], frameon=False)
        
        plot_idx += 1

plt.suptitle('2D Feature Space Visualizations', fontsize=14, y=0.95)  # Moved title higher
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout while preserving title space
plt.savefig(os.path.join(save_dir, '2D_feature_space.png'))
plt.show()

# Create 3D plots for all three-feature combinations
feature_combos = list(combinations(range(n_features), 3))
n_combos = len(feature_combos)
fig_3d = plt.figure(figsize=(15, 5*((n_combos+1)//2)))
plt.subplots_adjust(top=0.92)  # Increased space for title

for idx, (i, j, k) in enumerate(feature_combos):
    ax = fig_3d.add_subplot(((n_combos+1)//2), 2, idx+1, projection='3d')
    
    for site in sites:
        site_data = df[df['Site'] == site]
        ax.scatter(site_data[bp_names[i]], 
                  site_data[bp_names[j]], 
                  site_data[bp_names[k]],
                  label=site if idx == 0 else None,
                  alpha=0.6)
    
    ax.set_xlabel(bp_names[i], fontsize=10)
    ax.set_ylabel(bp_names[j], fontsize=10)
    ax.set_zlabel(bp_names[k], fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.view_init(elev=20, azim=45)
    
    if idx == 0:
        ax.legend(bbox_to_anchor=(1.15, 1))

plt.suptitle('3D Feature Space Visualizations', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '3D_feature_space.png'))
plt.show()
