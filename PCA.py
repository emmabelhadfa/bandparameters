import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import h5py
from scipy.stats import ttest_ind
from itertools import combinations

# Create save directory
save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters_new/pca'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Directory containing HDF5 files
data_dir = '/Users/emmabelhadfa/Documents/Oxford/orex/OTES/data/locations'

# Initialize lists to store data
all_features = []
all_sites = []

def find_band_parameters(wavenumbers, spectrum):
    """Calculate all band parameters for a single spectrum"""
    # 1. Find CF (maximum between 1200-1000)
    cf_mask = (wavenumbers >= 1000) & (wavenumbers <= 1200)
    cf_region = spectrum[cf_mask]
    cf_waves = wavenumbers[cf_mask]
    cf_idx = np.argmax(cf_region)
    cf_x, cf_y = cf_waves[cf_idx], cf_region[cf_idx]
    
    # 2. Calculate spectral slope from max wavenumber to CF
    max_wave = np.max(wavenumbers)
    slope_mask = (wavenumbers >= cf_x) & (wavenumbers <= max_wave)
    slope_x = wavenumbers[slope_mask]
    slope_y = spectrum[slope_mask]
    
    if len(slope_x) >= 2:
        slope, intercept = np.polyfit(slope_x[::-1], slope_y[::-1], 1)
        y_pred = slope * slope_x[::-1] + intercept
        r_squared = np.corrcoef(slope_y[::-1], y_pred)[0,1]**2
        slope_params = (slope, r_squared)
    else:
        slope_params = (None, None)
    
    # 3. Find Restrahlen stretching mode (minimum between 1100-800)
    stretch_mask = (wavenumbers >= 800) & (wavenumbers <= 1100)
    stretch_region = spectrum[stretch_mask]
    stretch_waves = wavenumbers[stretch_mask]
    stretch_idx = np.argmin(stretch_region)
    stretch_x, stretch_y = stretch_waves[stretch_idx], stretch_region[stretch_idx]
    
    # 4. Find Restrahlen bending mode (minimum between 600-400)
    bend_mask = (wavenumbers >= 400) & (wavenumbers <= 600)
    bend_region = spectrum[bend_mask]
    bend_waves = wavenumbers[bend_mask]
    bend_idx = np.argmin(bend_region)
    bend_x, bend_y = bend_waves[bend_idx], bend_region[bend_idx]
    
    return np.array([cf_x, slope_params[0], stretch_x, bend_x])

# Read data from HDF5 files
for file in os.listdir(data_dir):
    if file.endswith('.hdf5'):
        site_name = file.split('.')[0]
        file_path = os.path.join(data_dir, file)
        
        with h5py.File(file_path, "r") as f:
            mt_emissivity = f["mt_emissivity"][:, :, :]
            xaxis_L3 = f["xaxis_L3"][:, 0, 0]
            
            print(f"Processing {site_name}")
            
            # Remove outliers exactly like emissivityplot.py
            mt_emissivity_clipped = np.clip(mt_emissivity, -1.2, 1.2)
            means = np.mean(mt_emissivity_clipped, axis=(0, 1))
            z_scores = (means - np.mean(means)) / np.std(means)
            non_outliers = np.where(np.abs(z_scores) <= 3)[0]
            
            # Get clean data
            non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers[non_outliers < mt_emissivity_clipped.shape[2]]]
            
            # Process each spectrum
            for i in range(non_outlier_emissivity.shape[2]):
                spectrum = non_outlier_emissivity[:, 0, i]
                features = find_band_parameters(xaxis_L3, spectrum)
                
                if not any(np.isnan(features)):
                    all_features.append(features)
                    all_sites.append(site_name)

# Convert to numpy arrays
X = np.array(all_features)
sites = np.array(all_sites)
feature_names = ['CF Position', 'Spectral Slope', 'Stretch Position', 'Bend Position']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute PCA
U, S, Vt = np.linalg.svd(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = S**2 / np.sum(S**2)

# Get principal components
pc_scores = U[:, :2] * S[:2]

# Calculate distance matrix
unique_sites = np.unique(sites)
n_sites = len(unique_sites)
distance_matrix = np.zeros((n_sites, n_sites))

# Calculate pairwise distances between sites
for i, site1 in enumerate(unique_sites):
    for j, site2 in enumerate(unique_sites[i+1:], i+1):
        site1_mask = sites == site1
        site2_mask = sites == site2
        
        site1_scores = pc_scores[site1_mask]
        site2_scores = pc_scores[site2_mask]
        
        # Calculate Euclidean distance between site centroids in PC space
        centroid_dist = np.linalg.norm(np.mean(site1_scores, axis=0) - np.mean(site2_scores, axis=0))
        distance_matrix[i, j] = centroid_dist
        distance_matrix[j, i] = centroid_dist

# Save results to text file
with open(os.path.join(save_dir, 'pca_analysis_results.txt'), 'w') as f:
    f.write("PCA Analysis Results\n")
    f.write("===================\n\n")
    
    f.write("Dataset Summary\n")
    f.write("--------------\n")
    f.write(f"Total number of valid spectra: {len(X)}\n")
    f.write("Number of spectra per site:\n")
    for site in unique_sites:
        f.write(f"{site}: {np.sum(sites == site)}\n")
    
    # Add explained variance information
    f.write("\nExplained Variance by Principal Component\n")
    f.write("-------------------------------------\n")
    for i, var in enumerate(explained_variance_ratio):
        f.write(f"PC{i+1}: {var*100:.1f}%\n")
    
    # Add correlation matrix
    f.write("\nFeature Correlation Matrix\n")
    f.write("----------------------\n")
    f.write(f"Features: {', '.join(feature_names)}\n")
    corr_matrix = np.corrcoef(X_scaled.T)
    for i in range(len(feature_names)):
        f.write(f"{feature_names[i]}: " + " ".join(f"{corr_matrix[i,j]:6.3f}" for j in range(len(feature_names))) + "\n")
    
    # Add statistical analysis results
    f.write("\nStatistical Analysis\n")
    f.write("-------------------\n")
    for site in unique_sites:
        site_mask = sites == site
        site_data = X_scaled[site_mask]
        f.write(f"\n{site} Statistics:\n")
        for i, feature in enumerate(feature_names):
            mean = np.mean(site_data[:, i])
            std = np.std(site_data[:, i])
            f.write(f"{feature:>8}: mean = {mean:.3f}, std = {std:.3f}\n")
    
    # Add pairwise site comparisons with t-tests
    f.write("\nPairwise Site Comparisons\n")
    f.write("----------------------\n")
    for i, site1 in enumerate(unique_sites):
        for j, site2 in enumerate(unique_sites[i+1:], i+1):
            site1_mask = sites == site1
            site2_mask = sites == site2
            site1_data = X_scaled[site1_mask]
            site2_data = X_scaled[site2_mask]
            
            f.write(f"\n{site1} vs {site2}:\n")
            f.write(f"PC Space Distance: {distance_matrix[i,j]:.3f}\n")
            
            # Overall comparison using all features
            try:
                # Calculate mean difference vector
                mean_diff = np.mean(site1_data, axis=0) - np.mean(site2_data, axis=0)
                
                # Calculate pooled covariance matrix
                n1, n2 = len(site1_data), len(site2_data)
                pooled_cov = ((n1-1)*np.cov(site1_data.T) + (n2-1)*np.cov(site2_data.T)) / (n1+n2-2)
                
                # Calculate Hotelling's T-squared statistic
                t_squared = (n1*n2)/(n1+n2) * mean_diff.dot(np.linalg.pinv(pooled_cov)).dot(mean_diff)
                
                # Convert to F-statistic
                p = len(mean_diff)  # number of variables
                f_stat = ((n1+n2-p-1)/(p*(n1+n2-2))) * t_squared
                
                # Calculate p-value using scipy.stats.f
                from scipy.stats import f as f_dist  # Rename the import
                p_val = 1 - f_dist.cdf(f_stat, p, n1+n2-p-1)
                
                f.write("\nOverall Comparison (Hotelling's T-squared test):\n")
                f.write(f"T-squared = {t_squared:.3f}, F({p},{n1+n2-p-1}) = {f_stat:.3f}, p = {p_val:.3e}\n")
                
                # Add interpretation of significance
                if p_val < 0.001:
                    significance = "Highly significant"
                elif p_val < 0.01:
                    significance = "Very significant"
                elif p_val < 0.05:
                    significance = "Significant"
                else:
                    significance = "Not significant"
                f.write(f"Significance: {significance} (α = 0.05)\n")
                
            except Exception as e:
                f.write("\nOverall Comparison: Error in calculation\n")
                f.write(f"Error details: {str(e)}\n")
            
            # Individual feature t-tests
            f.write("\nFeature-wise t-tests:\n")
            for k, feature in enumerate(feature_names):
                try:
                    # Check if there's enough variation in both groups
                    if np.std(site1_data[:, k]) > 0 and np.std(site2_data[:, k]) > 0:
                        t_stat, p_val = ttest_ind(site1_data[:, k], site2_data[:, k], equal_var=False)
                        if np.isnan(t_stat) or np.isnan(p_val):
                            f.write(f"{feature:>8}: Insufficient variation for t-test\n")
                        else:
                            f.write(f"{feature:>8}: t = {t_stat:6.3f}, p = {p_val:.3e}\n")
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
                    else:
                        f.write(f"{feature:>8}: No variation in one or both groups\n")
                except Exception as e:
                    f.write(f"{feature:>8}: Error in t-test calculation: {str(e)}\n")

# Create color map for sites
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_sites)))
site_to_color = dict(zip(unique_sites, colors))

# Create pairwise plots for individual spectra (2D)
fig_ind = plt.figure(figsize=(12, 12))
plt.subplots_adjust(top=0.85)
n_features = len(feature_names)
plot_idx = 1

for i in range(n_features):
    for j in range(i+1, n_features):
        plt.subplot(n_features-1, n_features-1, plot_idx)
        
        for site in unique_sites:
            mask = sites == site
            plt.scatter(X[mask, i], X[mask, j],  # Using X instead of X_scaled
                       c=[site_to_color[site]], 
                       alpha=0.7,
                       label=site if plot_idx == 1 else "")
        
        plt.xlabel(f"{feature_names[i]} (cm⁻¹)" if i != 1 else "Spectral Slope", fontsize=10)
        plt.ylabel(f"{feature_names[j]} (cm⁻¹)" if j != 1 else "Spectral Slope", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if plot_idx == 1:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plot_idx += 1

plt.suptitle('Pairwise Feature Plots (Individual Spectra)', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pairwise_plots_individual.png'))
plt.close()

# Create 3D pairwise plots for individual spectra
feature_combos = list(combinations(range(n_features), 3))
n_combos = len(feature_combos)
fig_3d_ind = plt.figure(figsize=(20, 8*((n_combos+1)//2)))  # Increased height
plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.5, wspace=0.3)  # Adjusted margins

for idx, (i, j, k) in enumerate(feature_combos):
    ax = fig_3d_ind.add_subplot(((n_combos+1)//2), 2, idx+1, projection='3d')
    
    for site in unique_sites:
        mask = sites == site
        ax.scatter(X[mask, i], X[mask, j], X[mask, k],
                  c=[site_to_color[site]], 
                  alpha=0.6,
                  label=site if idx == 0 else "")
    
    ax.set_xlabel(f"{feature_names[i]} (cm⁻¹)" if i != 1 else "Spectral Slope", fontsize=10)
    ax.set_ylabel(f"{feature_names[j]} (cm⁻¹)" if j != 1 else "Spectral Slope", fontsize=10)
    ax.set_zlabel(f"{feature_names[k]} (cm⁻¹)" if k != 1 else "Spectral Slope", fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.view_init(elev=20, azim=45)
    
    # Make plot more cubic
    ax.set_box_aspect([1,1,1])
    
    # Adjust label padding
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
    
    if idx == 0:
        ax.legend(bbox_to_anchor=(1.15, 1))

plt.suptitle('3D Feature Space Visualizations (Individual Spectra)', fontsize=14, y=0.95)
plt.savefig(os.path.join(save_dir, '3D_pairwise_individual.png'), 
            bbox_inches='tight', 
            pad_inches=0.5)  # Added padding
plt.close()

# Create biplot for individual spectra
plt.figure(figsize=(12, 10))

# Plot scores (sites)
for site in unique_sites:
    mask = sites == site
    plt.scatter(pc_scores[mask, 0], pc_scores[mask, 1], 
                c=[site_to_color[site]], alpha=0.7, label=site)

# Plot loadings (feature vectors)
scaling_factor = np.max(np.abs(pc_scores)) / np.max(np.abs(Vt[:2, :]))
for i, feature in enumerate(feature_names):
    x_end = Vt[0, i] * scaling_factor
    y_end = Vt[1, i] * scaling_factor
    
    # Draw vector with thinner line
    plt.arrow(0, 0, x_end, y_end,
             color='red', alpha=0.5, width=0.004,
             head_width=0.02, head_length=0.02,
             length_includes_head=True)
    
    # Place label in middle of vector with white background
    plt.annotate(feature, 
                (x_end * 0.5, y_end * 0.5),
                color='darkred',
                fontsize=9,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% explained variance)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% explained variance)')
plt.title('PCA Biplot of Individual Spectra')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pca_biplot_individual.png'))
plt.close()  # Close the figure to free memory

# Create scree plot for individual spectra
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), 
         explained_variance_ratio * 100, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot (Individual Spectra)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(explained_variance_ratio) + 1))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'scree_plot_individual.png'))
plt.close()  # Close the figure to free memory

# Calculate site averages for averaged PCA
site_averages = []
site_labels = []
for site in unique_sites:
    mask = sites == site
    site_avg = np.mean(X[mask], axis=0)  # Calculate averages from original data
    site_averages.append(site_avg)
    site_labels.append(site)

X_avg = np.array(site_averages)
# Scale the site averages before PCA
scaler_avg = StandardScaler()
X_avg_scaled = scaler_avg.fit_transform(X_avg)

# Compute PCA on scaled averages
U_avg, S_avg, Vt_avg = np.linalg.svd(X_avg_scaled)
explained_variance_ratio_avg = S_avg**2 / np.sum(S_avg**2)
pc_scores_avg = U_avg[:, :2] * S_avg[:2]

# Create pairwise plots for site averages (2D)
fig_avg = plt.figure(figsize=(12, 12))
plt.subplots_adjust(top=0.85)
plot_idx = 1

for i in range(n_features):
    for j in range(i+1, n_features):
        plt.subplot(n_features-1, n_features-1, plot_idx)
        
        site_averages_unscaled = []
        for site in unique_sites:
            mask = sites == site
            site_avg = np.mean(X[mask], axis=0)  # Using X instead of X_scaled
            site_averages_unscaled.append(site_avg)
        
        X_avg_unscaled = np.array(site_averages_unscaled)
        plt.scatter(X_avg_unscaled[:, i], X_avg_unscaled[:, j], c=colors, alpha=0.7)
        for k, site in enumerate(site_labels):
            plt.annotate(site, (X_avg_unscaled[k, i], X_avg_unscaled[k, j]))
        
        plt.xlabel(f"{feature_names[i]} (cm⁻¹)" if i != 1 else "Spectral Slope", fontsize=10)
        plt.ylabel(f"{feature_names[j]} (cm⁻¹)" if j != 1 else "Spectral Slope", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plot_idx += 1

plt.suptitle('Pairwise Feature Plots (Site Averages)', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pairwise_plots_averaged.png'))
plt.close()

# Create 3D pairwise plots for site averages
fig_3d_avg = plt.figure(figsize=(20, 8*((n_combos+1)//2)))  # Increased height
plt.subplots_adjust(top=0.95, bottom=0.15, hspace=0.5, wspace=0.3)  # Adjusted margins

for idx, (i, j, k) in enumerate(feature_combos):
    ax = fig_3d_avg.add_subplot(((n_combos+1)//2), 2, idx+1, projection='3d')
    
    site_averages_unscaled = []
    for site in unique_sites:
        mask = sites == site
        site_avg = np.mean(X[mask], axis=0)
        site_averages_unscaled.append(site_avg)
    
    X_avg_unscaled = np.array(site_averages_unscaled)
    ax.scatter(X_avg_unscaled[:, i], X_avg_unscaled[:, j], X_avg_unscaled[:, k], 
              c=colors, alpha=0.7)
    
    for site_idx, site in enumerate(site_labels):
        ax.text(X_avg_unscaled[site_idx, i], 
                X_avg_unscaled[site_idx, j], 
                X_avg_unscaled[site_idx, k],
                site, fontsize=8)
    
    ax.set_xlabel(f"{feature_names[i]} (cm⁻¹)" if i != 1 else "Spectral Slope", fontsize=10)
    ax.set_ylabel(f"{feature_names[j]} (cm⁻¹)" if j != 1 else "Spectral Slope", fontsize=10)
    ax.set_zlabel(f"{feature_names[k]} (cm⁻¹)" if k != 1 else "Spectral Slope", fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.view_init(elev=20, azim=45)
    
    # Make plot more cubic
    ax.set_box_aspect([1,1,1])
    
    # Adjust label padding
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15

plt.suptitle('3D Feature Space Visualizations (Site Averages)', fontsize=14, y=0.95)
plt.savefig(os.path.join(save_dir, '3D_pairwise_averaged.png'), 
            bbox_inches='tight', 
            pad_inches=0.5)  # Added padding
plt.close()

# Create biplot for site averages
plt.figure(figsize=(12, 10))

# Plot scores (sites)
plt.scatter(pc_scores_avg[:, 0], pc_scores_avg[:, 1], c=colors, alpha=0.7)
for i, site in enumerate(site_labels):
    plt.annotate(site, (pc_scores_avg[i, 0], pc_scores_avg[i, 1]))

# Plot loadings with thinner lines
scaling_factor = np.max(np.abs(pc_scores_avg)) / np.max(np.abs(Vt_avg[:2, :]))
for i, feature in enumerate(feature_names):
    x_end = Vt_avg[0, i] * scaling_factor
    y_end = Vt_avg[1, i] * scaling_factor
    
    plt.arrow(0, 0, x_end, y_end,
             color='red', alpha=0.5, width=0.004,
             head_width=0.02, head_length=0.02,
             length_includes_head=True)
    
    plt.annotate(feature, 
                (x_end * 0.5, y_end * 0.5),
                color='darkred',
                fontsize=9,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))

plt.xlabel(f'PC1 ({explained_variance_ratio_avg[0]*100:.1f}% explained variance)')
plt.ylabel(f'PC2 ({explained_variance_ratio_avg[1]*100:.1f}% explained variance)')
plt.title('PCA Biplot of Site Averages')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pca_biplot_averaged.png'))
plt.close()  # Close the figure to free memory

# Create scree plot for averaged data
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio_avg) + 1), 
         explained_variance_ratio_avg * 100, 'bo-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Scree Plot (Site Averages)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(explained_variance_ratio_avg) + 1))
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'scree_plot_averaged.png'))
plt.close()  # Close the figure to free memory

# Calculate distance matrix for individual spectra
distance_matrix_ind = np.zeros((len(unique_sites), len(unique_sites)))

# Calculate pairwise distances between sites using individual spectra
for i, site1 in enumerate(unique_sites):
    for j, site2 in enumerate(unique_sites[i+1:], i+1):
        site1_mask = sites == site1
        site2_mask = sites == site2
        
        site1_scores = pc_scores[site1_mask]
        site2_scores = pc_scores[site2_mask]
        
        # Calculate Euclidean distance between site centroids in PC space
        centroid_dist = np.linalg.norm(np.mean(site1_scores, axis=0) - np.mean(site2_scores, axis=0))
        distance_matrix_ind[i, j] = centroid_dist
        distance_matrix_ind[j, i] = centroid_dist

# Create heatmap for individual spectra
plt.figure(figsize=(10, 8))
plt.imshow(distance_matrix_ind, cmap='cubehelix')
plt.colorbar(label='Euclidean Distance in PC Space')
plt.xticks(range(len(unique_sites)), unique_sites, rotation=45)
plt.yticks(range(len(unique_sites)), unique_sites)
plt.title('Pairwise Site Distances (Individual Spectra)')

for i in range(len(unique_sites)):
    for j in range(len(unique_sites)):
        plt.text(j, i, f'{distance_matrix_ind[i,j]:.2f}',
                ha='center', va='center',
                color='white' if distance_matrix_ind[i,j] < np.mean(distance_matrix_ind) else 'black')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'site_distances_heatmap_individual.png'))
plt.close()

# Calculate distance matrix for site averages
distance_matrix_avg = np.zeros((len(unique_sites), len(unique_sites)))

# Calculate pairwise distances between sites using averaged data
for i, site1 in enumerate(unique_sites):
    for j, site2 in enumerate(unique_sites[i+1:], i+1):
        site1_idx = site_labels.index(site1)
        site2_idx = site_labels.index(site2)
        
        # Calculate Euclidean distance between sites in PC space
        centroid_dist = np.linalg.norm(pc_scores_avg[site1_idx] - pc_scores_avg[site2_idx])
        distance_matrix_avg[i, j] = centroid_dist
        distance_matrix_avg[j, i] = centroid_dist

# Create heatmap for site averages
plt.figure(figsize=(10, 8))
plt.imshow(distance_matrix_avg, cmap='cubehelix')
plt.colorbar(label='Euclidean Distance in PC Space')
plt.xticks(range(len(unique_sites)), unique_sites, rotation=45)
plt.yticks(range(len(unique_sites)), unique_sites)
plt.title('Pairwise Site Distances (Site Averages)')

for i in range(len(unique_sites)):
    for j in range(len(unique_sites)):
        plt.text(j, i, f'{distance_matrix_avg[i,j]:.2f}',
                ha='center', va='center',
                color='white' if distance_matrix_avg[i,j] < np.mean(distance_matrix_avg) else 'black')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'site_distances_heatmap_averaged.png'))
plt.close()
