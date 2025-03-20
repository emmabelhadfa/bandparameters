import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneOut, cross_val_score
import h5py
import os
from sklearn.metrics import classification_report
from itertools import combinations
import mpl_toolkits.mplot3d
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, binomtest

# Add feature names definition
feature_names = ['CF Position', 'Spectral Slope', 'Stretch Position', 'Bend Position']

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

# Directory containing HDF5 files
data_dir = '/Users/emmabelhadfa/Documents/Oxford/orex/OTES/data/locations'

# Create save directory
save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters_new/knearest'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize lists to store data
individual_features = []
individual_sites = []
site_averages = {}

# Read data from HDF5 files
for file in os.listdir(data_dir):
    if file.endswith('.hdf5'):
        site_name = file.split('.')[0]
        file_path = os.path.join(data_dir, file)
        
        with h5py.File(file_path, "r") as f:
            mt_emissivity = f["mt_emissivity"][:, :, :]
            xaxis_L3 = f["xaxis_L3"][:, 0, 0]
            
            print(f"File: {file}")
            print(f"Emissivity shape: {mt_emissivity.shape}")
            print(f"Wavenumbers shape: {xaxis_L3.shape}")
            
            # 1. First remove outliers exactly like emissivityplot.py
            mt_emissivity_clipped = np.clip(mt_emissivity, -1.2, 1.2)
            means = np.mean(mt_emissivity_clipped, axis=(0, 1))
            z_scores = (means - np.mean(means)) / np.std(means)
            non_outliers = np.where(np.abs(z_scores) <= 3)[0]
            
            # Get clean data
            non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers[non_outliers < mt_emissivity_clipped.shape[2]]]
            site_average = np.mean(non_outlier_emissivity, axis=2)[:, 0]
            
            # 2. Now calculate band parameters for each clean observation
            for i in range(non_outlier_emissivity.shape[2]):
                spectrum = non_outlier_emissivity[:, 0, i]
                features = find_band_parameters(xaxis_L3, spectrum)
                if not np.any(np.isnan(features)):
                    individual_features.append(features)
                    individual_sites.append(site_name)
            
            # 3. Calculate parameters for the averaged spectrum
            avg_features = find_band_parameters(xaxis_L3, site_average)
            if not np.any(np.isnan(avg_features)):
                site_averages[site_name] = avg_features

# Convert to numpy arrays
X_individual = np.array(individual_features)
y_individual = np.array(individual_sites)
X_average = np.array(list(site_averages.values()))
y_average = np.array(list(site_averages.keys()))

if len(X_individual) == 0 and len(X_average) == 0:
    print("No valid spectra found!")
    exit()

print(f"\nTotal number of valid individual spectra: {len(X_individual)}")
print(f"Total number of valid site averages: {len(X_average)}")
print(f"Number of individual spectra per site:")
for site in np.unique(y_individual):
    print(f"{site}: {np.sum(y_individual == site)}")
print(f"Number of site averages per site:")
for site in np.unique(y_average):
    print(f"{site}: {np.sum(y_average == site)}")

# Standardize features
scaler = StandardScaler()
X_individual_scaled = scaler.fit_transform(X_individual)
X_average_scaled = scaler.transform(X_average)

# Apply PCA
pca = PCA()
X_individual_pca = pca.fit_transform(X_individual_scaled)
X_average_pca = pca.transform(X_average_scaled)

# Calculate explained variance ratio
explained_variance_individual = pca.explained_variance_ratio_ * 100
explained_variance_average = pca.explained_variance_ratio_ * 100

# Determine number of components to explain 95% of variance
cumulative_variance_individual = np.cumsum(explained_variance_individual)
cumulative_variance_average = np.cumsum(explained_variance_average)
n_components_individual = np.argmax(cumulative_variance_individual >= 95) + 1
n_components_average = np.argmax(cumulative_variance_average >= 95) + 1

# Use first n_components for KNN
X_individual_reduced = X_individual_pca[:, :n_components_individual]
X_average_reduced = X_average_pca[:, :n_components_average]

# Modify the KNN analysis section:
# For individual spectra - now using up to 10 neighbors
max_k_individual = min(10, len(X_individual_scaled) - 1)  # Changed back to 10
k_values_individual = range(1, max_k_individual + 1)
accuracies_individual = []

# For site averages - still using up to 3 neighbors since we have 4 sites
max_k_average = min(3, len(X_average_scaled) - 1)
k_values_average = range(1, max_k_average + 1)
accuracies_average = []

# Use LeaveOneOut cross-validation
loo = LeaveOneOut()

# Calculate accuracies for individual spectra
for k in k_values_individual:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_individual_scaled, y_individual, cv=loo)
    accuracies_individual.append(scores.mean())

# Calculate accuracies for site averages
for k in k_values_average:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_average_scaled, y_average, cv=loo)
    accuracies_average.append(scores.mean())

# Plot accuracy vs k value (individual spectra only)
plt.figure(figsize=(12, 6))
plt.plot(k_values_individual, accuracies_individual, 'bo-', linewidth=2, label='Individual Spectra')
plt.xlabel('Number of Neighbors (k)', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.title('KNN Classification Accuracy vs k', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.xticks(range(1, max_k_individual + 1))  # Changed back to show all k values

# Add accuracy labels
for k, acc in zip(k_values_individual, accuracies_individual):
    plt.text(k, acc + 0.01, f'{acc:.2f}', 
             ha='center', va='bottom', color='blue')

plt.legend(loc='lower right', framealpha=0.9, edgecolor='black')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'accuracy_vs_k.png'))
plt.show()

# Define color mapping for sites (add this before the plotting sections)
# Use the same rainbow colormap as in PCA.py
colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(y_individual))))
site_to_color = dict(zip(np.unique(y_individual), colors))

# Create heatmaps for both datasets
for analysis_type, X, y, labels in [
    ("individual", X_individual_scaled, y_individual, np.unique(y_individual)),
    ("average", X_average_scaled, y_average, np.unique(y_average))
]:
    # Calculate confusion matrix-like distances
    distance_matrix = np.zeros((len(labels), len(labels)))
    
    for i, site1 in enumerate(labels):
        for j, site2 in enumerate(labels[i+1:], i+1):
            site1_mask = y == site1
            site2_mask = y == site2
            site1_data = X[site1_mask]
            site2_data = X[site2_mask]
            
            # Calculate average pairwise distance between sites
            if analysis_type == "individual":
                # For individual spectra, calculate mean pairwise distance
                distances = []
                for point1 in site1_data:
                    for point2 in site2_data:
                        dist = np.linalg.norm(point1 - point2)
                        distances.append(dist)
                avg_dist = np.mean(distances)
            else:
                # For site averages, use direct distance between means
                avg_dist = np.linalg.norm(np.mean(site1_data, axis=0) - np.mean(site2_data, axis=0))
            
            distance_matrix[i, j] = avg_dist
            distance_matrix[j, i] = avg_dist
    
    plt.figure(figsize=(10, 8))
    plt.imshow(distance_matrix, cmap='cubehelix')
    plt.colorbar(label='Average Distance')
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.title(f'Pairwise Site Distances ({analysis_type.title()} Spectra)')
    
    mean_dist = np.mean(distance_matrix)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f'{distance_matrix[i,j]:.2f}',
                    ha='center', va='center',
                    color='white' if distance_matrix[i,j] < mean_dist else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'site_distances_heatmap_{analysis_type}.png'))
    plt.show()

# Save results to text file
with open(os.path.join(save_dir, 'knn_analysis_results.txt'), 'w') as f:
    f.write("KNN Analysis Results\n")
    f.write("===================\n\n")
    
    f.write("Dataset Summary\n")
    f.write("--------------\n")
    f.write(f"Total number of valid individual spectra: {len(X_individual)}\n")
    f.write(f"Total number of valid site averages: {len(X_average)}\n")
    f.write("\nNumber of individual spectra per site:\n")
    for site in np.unique(y_individual):
        f.write(f"{site}: {np.sum(y_individual == site)}\n")
    
    # Add classification reports for both datasets
    f.write("\nClassification Report (Individual Spectra)\n")
    f.write("-------------------------------------\n")
    knn = KNeighborsClassifier(n_neighbors=3)  # Using k=3 as a reasonable default
    knn.fit(X_individual_scaled, y_individual)
    y_pred_individual = knn.predict(X_individual_scaled)
    f.write(classification_report(y_individual, y_pred_individual, zero_division=0))
    
    f.write("\nClassification Report (Site Averages)\n")
    f.write("--------------------------------\n")
    knn_avg = KNeighborsClassifier(n_neighbors=1)  # k=1 for site averages due to small sample size
    knn_avg.fit(X_average_scaled, y_average)
    y_pred_average = knn_avg.predict(X_average_scaled)
    f.write(classification_report(y_average, y_pred_average, zero_division=0))
    
    # Add statistical analysis for both datasets
    f.write("\nStatistical Analysis (Individual Spectra)\n")
    f.write("-------------------------------------\n")
    for site in np.unique(y_individual):
        site_mask = y_individual == site
        site_data = X_individual[site_mask]
        f.write(f"\n{site} Statistics:\n")
        for i, feature in enumerate(feature_names):
            mean = np.mean(site_data[:, i])
            std = np.std(site_data[:, i])
            f.write(f"{feature:>8}: mean = {mean:.3f}, std = {std:.3f}\n")
    
    f.write("\nStatistical Analysis (Site Averages)\n")
    f.write("--------------------------------\n")
    for site in np.unique(y_average):
        site_mask = y_average == site
        site_data = X_average[site_mask]
        f.write(f"\n{site} Statistics:\n")
        for i, feature in enumerate(feature_names):
            mean = np.mean(site_data[:, i])
            std = np.std(site_data[:, i])
            f.write(f"{feature:>8}: mean = {mean:.3f}, std = {std:.3f}\n")
