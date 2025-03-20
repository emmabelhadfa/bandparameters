import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations
import seaborn as sns
from scipy.stats import ttest_ind

# Create save directory
save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters/ratios'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Directory containing HDF5 files
data_directory = "/Users/emmabelhadfa/Documents/Oxford/orex/OTES/data/locations/"

# Get all HDF5 file paths in the directory
file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".hdf5")]

# Create figure with 3 subplots (original 2 plus new 3D scatter)
fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(311)  # Emissivity plot
ax2 = fig.add_subplot(312)  # Bar plot
ax3 = fig.add_subplot(313, projection='3d')  # New 3D scatter plot

# Define band parameter regions based on PCA.py
bp1_region = (950, 1150)  # BP1 region
bp2_region = (800, 950)   # BP2 region 
bp4_region = (400, 500)   # BP4 region

# Define colors for band parameter regions
bp_colors = ['red', 'green', 'blue']  # Simple color names, alpha will be set in axvspan

ratios_by_site = {}
site_spectra = {}
all_ratios = []
all_sites = []

# Process each file
for file_path in file_paths:
    site_name = os.path.basename(file_path).split('.')[0]
    
    with h5py.File(file_path, "r") as file:
        mt_emissivity = file["mt_emissivity"][:, :, :]
        xaxis_L3 = file["xaxis_L3"][:, 0, 0]

    print(f"Processing {site_name}")
    print(f"Emissivity shape: {mt_emissivity.shape}")
    print(f"Wavenumbers shape: {xaxis_L3.shape}")

    # Transpose the emissivity data to get spectra
    mt_emissivity_transposed = mt_emissivity.transpose(2, 0, 1)  # Now it's (N, 208, 1)
    mt_emissivity_clipped = np.clip(mt_emissivity_transposed, -1.2, 1.2)
    
    # Process each spectrum
    for i in range(mt_emissivity_clipped.shape[0]):  # Loop over N spectra
        spectrum = mt_emissivity_clipped[i, :, 0]  # Get a single spectrum (208 points)
        
        # Calculate z-scores for outlier detection
        z_scores = (spectrum - np.mean(spectrum)) / np.std(spectrum)
        if np.all(np.abs(z_scores) <= 3):  # Only use non-outlier spectra
            # Calculate band parameters
            bp_values = []
            for region in [bp1_region, bp2_region, bp4_region]:
                mask = (xaxis_L3 >= region[0]) & (xaxis_L3 <= region[1])
                bp_value = np.mean(spectrum[mask])
                bp_values.append(bp_value)
            
            # Calculate ratios
            ratios = {
                'BP1/BP2': bp_values[0] / bp_values[1],
                'BP1/BP4': bp_values[0] / bp_values[2],
                'BP2/BP4': bp_values[1] / bp_values[2]
            }
            all_ratios.append(list(ratios.values()))
            all_sites.append(site_name)
            
            if site_name not in ratios_by_site:
                ratios_by_site[site_name] = []
            ratios_by_site[site_name].append(ratios)

# Convert to numpy arrays
X = np.array(all_ratios)
sites = np.array(all_sites)
ratio_names = ['BP1/BP2', 'BP1/BP4', 'BP2/BP4']

# Calculate distance matrix
unique_sites = np.unique(sites)
n_sites = len(unique_sites)
distance_matrix = np.zeros((n_sites, n_sites))

# Calculate pairwise distances between sites
for i, site1 in enumerate(unique_sites):
    for j, site2 in enumerate(unique_sites[i+1:], i+1):
        site1_data = X[sites == site1]
        site2_data = X[sites == site2]
        
        # Calculate Euclidean distance between site centroids
        centroid_dist = np.linalg.norm(np.mean(site1_data, axis=0) - np.mean(site2_data, axis=0))
        distance_matrix[i, j] = centroid_dist
        distance_matrix[j, i] = centroid_dist

# Save results to text file
with open(os.path.join(save_dir, 'ratio_analysis_results.txt'), 'w') as f:
    f.write("Band Ratio Analysis Results\n")
    f.write("=========================\n\n")
    
    f.write("Dataset Summary\n")
    f.write("--------------\n")
    f.write(f"Total number of spectra: {len(X)}\n")
    f.write("Number of spectra per site:\n")
    for site in unique_sites:
        f.write(f"{site}: {np.sum(sites == site)}\n")
    
    f.write("\nStatistical Analysis\n")
    f.write("-------------------\n")
    for site in unique_sites:
        site_data = X[sites == site]
        f.write(f"\n{site} Statistics:\n")
        for i, ratio in enumerate(ratio_names):
            mean = np.mean(site_data[:, i])
            std = np.std(site_data[:, i])
            f.write(f"{ratio:>8}: mean = {mean:.3f}, std = {std:.3f}\n")
    
    # Add pairwise t-tests with significance levels
    f.write("\nPairwise Site Comparisons\n")
    f.write("----------------------\n")
    for i, site1 in enumerate(unique_sites):
        for j, site2 in enumerate(unique_sites[i+1:], i+1):
            site1_data = X[sites == site1]
            site2_data = X[sites == site2]
            
            f.write(f"\n{site1} vs {site2}:\n")
            f.write(f"Centroid Distance: {distance_matrix[i,j]:.3f}\n")
            
            for k, ratio in enumerate(ratio_names):
                t_stat, p_val = ttest_ind(site1_data[:, k], site2_data[:, k])
                f.write(f"{ratio:>8}: t = {t_stat:6.3f}, p = {p_val:.3e}\n")
                
                # Add interpretation of significance
                if p_val < 0.001:
                    significance = "Highly significant"
                elif p_val < 0.01:
                    significance = "Very significant"
                elif p_val < 0.05:
                    significance = "Significant"
                else:
                    significance = "Not significant"
                f.write(f"{' '*9}Significance: {significance} (α = 0.05)\n")

# Create 2D subplot figure for all ratio combinations
n_ratios = len(ratio_names)
fig_2d = plt.figure(figsize=(15, 15))
plt.subplots_adjust(top=0.82)  # Increased space for title
plot_idx = 1

# Create numeric labels for coloring
label_to_num = {label: i for i, label in enumerate(unique_sites)}
numeric_labels = np.array([label_to_num[label] for label in sites])

for i in range(n_ratios):
    for j in range(i+1, n_ratios):
        plt.subplot(n_ratios-1, n_ratios-1, plot_idx)
        
        scatter = plt.scatter(X[:, i], X[:, j], 
                            c=numeric_labels, 
                            cmap=plt.cm.rainbow, 
                            edgecolors='black',
                            alpha=0.5)
        
        plt.xlabel(ratio_names[i], fontsize=10)
        plt.ylabel(ratio_names[j], fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        plot_idx += 1

plt.suptitle('2D Band Ratio Space Visualizations', fontsize=14, y=0.98)
plt.tight_layout()

# Add legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=plt.cm.rainbow(i/len(unique_sites)), 
                            label=site, markersize=10)
                 for i, site in enumerate(unique_sites)]
fig_2d.legend(handles=legend_elements, 
              title="Sites",
              loc='center right',
              bbox_to_anchor=(0.98, 0.5))
plt.savefig(os.path.join(save_dir, '2D_ratio_space.png'))
plt.show()

# Create 3D plots for all three-feature combinations
feature_combos = list(combinations(range(n_ratios), 3))
n_combos = len(feature_combos)
fig_3d = plt.figure(figsize=(15, 5*((n_combos+1)//2)))
plt.subplots_adjust(top=0.92)  # Increased space for title

for idx, (i, j, k) in enumerate(feature_combos):
    ax = fig_3d.add_subplot(((n_combos+1)//2), 2, idx+1, projection='3d')
    
    scatter = ax.scatter(X[:, i], X[:, j], X[:, k],
                        c=numeric_labels, 
                        cmap=plt.cm.rainbow,
                        edgecolors='black',
                        alpha=0.6)
    
    ax.set_xlabel(ratio_names[i], fontsize=10)
    ax.set_ylabel(ratio_names[j], fontsize=10)
    ax.set_zlabel(ratio_names[k], fontsize=10)
    ax.grid(True, alpha=0.4)
    
    # Rotate for better visibility
    ax.view_init(elev=20, azim=45)

plt.suptitle('3D Band Ratio Space Visualizations', fontsize=14, y=0.98)
plt.tight_layout()

# Add legend
fig_3d.legend(handles=legend_elements, 
              title="Sites",
              loc='center right',
              bbox_to_anchor=(0.98, 0.5))
plt.savefig(os.path.join(save_dir, '3D_ratio_space.png'))
plt.show()

# Create heatmap of site distances
plt.figure(figsize=(10, 8))
plt.imshow(distance_matrix, cmap='viridis')
plt.colorbar(label='Euclidean Distance')
plt.xticks(range(n_sites), unique_sites, rotation=45)
plt.yticks(range(n_sites), unique_sites)
plt.title('Pairwise Site Distances (Band Ratios)')

for i in range(n_sites):
    for j in range(n_sites):
        plt.text(j, i, f'{distance_matrix[i,j]:.4f}',  # Increased to 4 decimal places
                ha='center', va='center',
                color='white' if distance_matrix[i,j] > np.mean(distance_matrix) else 'black')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'site_distances_heatmap.png'))
plt.show()

# Plot original spectra
for site_name in ratios_by_site.keys():
    # Add vertical lines for band parameter regions with different colors
    for region, color in zip([bp1_region, bp2_region, bp4_region], bp_colors):
        ax1.axvspan(region[0], region[1], alpha=0.2, color=color, label=f'BP{color}')

ax1.set_xlabel("Wavenumber (cm⁻¹)")
ax1.set_ylabel("Emissivity")
ax1.set_title("Band Parameter Regions")
ax1.set_xlim(1500, 300)
ax1.set_ylim(0.95, 0.99)
ax1.grid(True)
ax1.legend(['BP1', 'BP2', 'BP4'])

# Plot band ratios
sites = list(ratios_by_site.keys())
ratio_types = ['BP1/BP2', 'BP1/BP4', 'BP2/BP4']
x = np.arange(len(sites))
width = 0.25

# Calculate average ratios for each site
site_avg_ratios = {}
for site in sites:
    site_ratios = ratios_by_site[site]  # List of dictionaries
    avg_ratios = {}
    for ratio_type in ratio_types:
        values = [r[ratio_type] for r in site_ratios]  # Extract values for this ratio type
        avg_ratios[ratio_type] = np.mean(values)
    site_avg_ratios[site] = avg_ratios

# Create 3D scatter plot
fig_3d = plt.figure(figsize=(12, 8))
ax3 = fig_3d.add_subplot(111, projection='3d')

# Plot each site's data points
for site in sites:
    site_ratios = ratios_by_site[site]
    bp1_bp2 = [r['BP1/BP2'] for r in site_ratios]
    bp1_bp4 = [r['BP1/BP4'] for r in site_ratios]
    bp2_bp4 = [r['BP2/BP4'] for r in site_ratios]
    
    ax3.scatter(bp1_bp2, bp1_bp4, bp2_bp4, 
                label=site, alpha=0.6)

ax3.set_xlabel('BP1/BP2')
ax3.set_ylabel('BP1/BP4')
ax3.set_zlabel('BP2/BP4')
ax3.set_title('3D Visualization of Band Ratios')

# Add legend
ax3.legend(title="Sites")

# Adjust the view angle for better visualization
ax3.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, '3D_ratio_space.png'))
plt.show()

# Print numerical results
print("\nBand Parameter Ratios by Site:")
for site in sites:
    print(f"\n{site}:")
    # Calculate average ratios for this site
    site_ratios = ratios_by_site[site]  # List of dictionaries
    avg_ratios = {}
    for ratio_type in ratio_names:
        values = [r[ratio_type] for r in site_ratios]
        avg = np.mean(values)
        std = np.std(values)
        print(f"{ratio_type}: mean = {avg:.3f}, std = {std:.3f}")

