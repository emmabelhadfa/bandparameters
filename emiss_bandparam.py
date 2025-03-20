import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Directory containing HDF5 files
data_directory = "/Users/emmabelhadfa/Documents/Oxford/orex/OTES/data/locations/"
save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters_new/emissivity'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Get all HDF5 file paths in the directory
file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".hdf5")]

# Create figures (after imports, before processing)
fig_avg, ax_avg = plt.subplots(figsize=(12, 6))  # For averaged spectra
fig_dist, axs_dist = plt.subplots(2, 2, figsize=(12, 10))  # For parameter distributions
axs_dist = axs_dist.flatten()

# Process each file
site_data = {}

def find_flexion_point(x, y, range_min, range_max):
    # Get data within range
    mask = (x >= range_min) & (x <= range_max)
    x_range = x[mask]
    y_range = y[mask]
    
    # Calculate second derivative using gradient twice
    first_derivative = np.gradient(y_range, x_range)
    second_derivative = np.gradient(first_derivative, x_range)
    
    # Find where second derivative crosses zero
    zero_crossings = np.where(np.diff(np.signbit(second_derivative)))[0]
    
    if len(zero_crossings) > 0:
        # Return the x value at the flexion point
        idx = zero_crossings[len(zero_crossings)//2]  # Take middle point if multiple exist
        return x_range[idx], y_range[idx]
    return None, None

def find_cf(wavenumbers, spectrum):
    """Find Christiansen Feature (emissivity maximum at wavenumber > 1000)"""
    mask = wavenumbers > 1000
    if not any(mask):
        return None, None
    cf_idx = np.argmax(spectrum[mask]) + np.where(mask)[0][0]
    return wavenumbers[cf_idx], spectrum[cf_idx]

def find_transparency_feature(wavenumbers, spectrum):
    """Find start and end of transparency feature (broad emissivity minima)"""
    mask = (wavenumbers >= 800) & (wavenumbers <= 1000)
    if not any(mask):
        return None, None
    
    region = spectrum[mask]
    w_region = wavenumbers[mask]
    slopes = np.diff(region)
    slope_changes = np.where(np.abs(np.diff(slopes)) > np.std(slopes))[0]
    
    if len(slope_changes) >= 2:
        start_idx = slope_changes[0] + np.where(mask)[0][0]
        end_idx = slope_changes[-1] + np.where(mask)[0][0]
        return (wavenumbers[start_idx], wavenumbers[end_idx])
    return None, None

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
        # Fit from high to low wavenumbers to get positive slope
        slope, intercept = np.polyfit(slope_x[::-1], slope_y[::-1], 1)
        # Calculate R-squared
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
    
    return {
        'CF': (cf_x, cf_y),
        'Slope': slope_params,
        'Stretch': (stretch_x, stretch_y),
        'Bend': (bend_x, bend_y)
    }

for file_path in file_paths:
    site_name = os.path.basename(file_path).split('.')[0]
    
    with h5py.File(file_path, "r") as file:
        mt_emissivity = file["mt_emissivity"][:, :, :]
        xaxis_L3 = file["xaxis_L3"][:, 0, 0]

    # 1. First remove outliers exactly like emissivityplot.py
    mt_emissivity_clipped = np.clip(mt_emissivity, -1.2, 1.2)
    means = np.mean(mt_emissivity_clipped, axis=(0, 1))
    z_scores = (means - np.mean(means)) / np.std(means)
    non_outliers = np.where(np.abs(z_scores) <= 3)[0]
    
    # Get clean data
    non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers[non_outliers < mt_emissivity_clipped.shape[2]]]
    site_average = np.mean(non_outlier_emissivity, axis=2)[:, 0]
    
    # 2. Now calculate band parameters for each clean observation
    individual_params = []
    for i in range(non_outlier_emissivity.shape[2]):
        spectrum = non_outlier_emissivity[:, 0, i]
        params = find_band_parameters(xaxis_L3, spectrum)
        individual_params.append(params)
    
    # 3. Calculate parameters for the averaged spectrum
    avg_params = find_band_parameters(xaxis_L3, site_average)
    
    site_data[site_name] = {
        'xaxis': xaxis_L3,
        'average': site_average,
        'avg_params': avg_params,
        'individual_params': individual_params
    }

# 4. Create parameter distribution plot
fig_dist, axs_dist = plt.subplots(2, 2, figsize=(12, 10))
axs_dist = axs_dist.flatten()

param_names = ['CF', 'Slope', 'Stretch', 'Bend']
for idx, param_name in enumerate(param_names):
    data_to_plot = []
    labels = []
    
    for site_name, data in site_data.items():
        if param_name == 'Slope':
            values = [p['Slope'][0] for p in data['individual_params'] if p['Slope'][0] is not None]
            # Normalize slopes by dividing by the mean slope across all sites
            if len(values) > 0:
                data_to_plot.append(values)
        else:
            values = [p[param_name][0] for p in data['individual_params'] if p[param_name][0] is not None]
            data_to_plot.append(values)
        labels.append(site_name)
    
    sns.boxplot(data=data_to_plot, ax=axs_dist[idx])
    axs_dist[idx].set_xticklabels(labels, rotation=45)
    if param_name == 'Slope':
        # Calculate relative slopes
        all_slopes = np.concatenate(data_to_plot)
        mean_slope = np.mean(all_slopes)
        normalized_data = [np.array(slopes)/mean_slope for slopes in data_to_plot]
        axs_dist[idx].clear()  # Clear previous plot
        sns.boxplot(data=normalized_data, ax=axs_dist[idx])
        axs_dist[idx].set_xticklabels(labels, rotation=45)
        axs_dist[idx].set_ylabel('Relative Slope (normalized)')
        axs_dist[idx].axhline(y=1, color='r', linestyle='--', alpha=0.5)  # Add reference line at mean
    else:
        axs_dist[idx].set_ylabel('Wavenumber (cm⁻¹)')
    axs_dist[idx].set_title(f'{param_name} Distribution')
    axs_dist[idx].grid(True)

fig_dist.suptitle('Distribution of Band Parameters Across Sites')
fig_dist.tight_layout()

# 5. Create averaged spectra plot with band parameter labels
fig_avg, ax_avg = plt.subplots(figsize=(12, 6))

for site_name, data in site_data.items():
    color = plt.cm.tab10(list(site_data.keys()).index(site_name))
    
    # Plot average spectrum
    ax_avg.plot(data['xaxis'], data['average'], '-', 
                label=site_name, alpha=0.7, color=color)
    
    # Add band parameter markers with consistent offsets
    params = data['avg_params']
    
    # CF marker and label
    ax_avg.plot(params['CF'][0], params['CF'][1], 'o', color=color, markersize=8)
    ax_avg.annotate(f'CF ({params["CF"][0]:.0f})', 
                   (params['CF'][0], params['CF'][1]),
                   xytext=(10, 10), textcoords='offset points',
                   color=color, fontsize=8)
    
    # Stretch marker and label
    ax_avg.plot(params['Stretch'][0], params['Stretch'][1], 's', color=color, markersize=8)
    ax_avg.annotate(f'Stretch ({params["Stretch"][0]:.0f})',
                   (params['Stretch'][0], params['Stretch'][1]),
                   xytext=(10, -15), textcoords='offset points',
                   color=color, fontsize=8)
    
    # Bend marker and label
    ax_avg.plot(params['Bend'][0], params['Bend'][1], '^', color=color, markersize=8)
    ax_avg.annotate(f'Bend ({params["Bend"][0]:.0f})',
                   (params['Bend'][0], params['Bend'][1]),
                   xytext=(-10, -15), textcoords='offset points',
                   color=color, fontsize=8)

ax_avg.set_xlabel("Wavenumber (cm⁻¹)")
ax_avg.set_ylabel("Emissivity")
ax_avg.set_xlim(1400, 300)
ax_avg.set_ylim(0.95, 0.99)
ax_avg.grid(True)
ax_avg.legend()
ax_avg.set_title("Site-Averaged Spectra with Band Parameters")

# Save plots
fig_dist.savefig(os.path.join(save_dir, 'parameter_distributions.png'))
fig_avg.savefig(os.path.join(save_dir, 'averaged_spectra_with_parameters.png'))

# Save parameters to text files
for site_name, params in site_data.items():
    with open(os.path.join(save_dir, f'{site_name}_parameters.txt'), 'w') as f:
        f.write(f"Band Parameters for {site_name}\n")
        f.write("=" * (len(site_name) + 18) + "\n\n")
        
        # Write averaged parameters
        f.write("Averaged Spectrum Parameters:\n")
        f.write("-" * 26 + "\n")
        avg_params = params['avg_params']
        
        if avg_params['CF'][0] is not None:
            f.write(f"CF: {avg_params['CF'][0]:.2f} cm⁻¹, {avg_params['CF'][1]:.4f}\n")
        if avg_params['Slope'][0] is not None:
            f.write(f"Spectral Slope: {avg_params['Slope'][0]:.6f}, R² = {avg_params['Slope'][1]:.4f}\n")
        f.write(f"Restrahlen Stretch: {avg_params['Stretch'][0]:.2f} cm⁻¹, {avg_params['Stretch'][1]:.4f}\n")
        f.write(f"Restrahlen Bend: {avg_params['Bend'][0]:.2f} cm⁻¹, {avg_params['Bend'][1]:.4f}\n")
        
        # Write individual statistics
        f.write("\nIndividual Spectra Statistics:\n")
        f.write("-" * 27 + "\n")
        
        # Calculate statistics
        cf_values = [p['CF'][1] for p in params['individual_params'] if p['CF'][0] is not None]
        slope_values = [p['Slope'][0] for p in params['individual_params'] if p['Slope'][0] is not None]
        stretch_values = [p['Stretch'][1] for p in params['individual_params']]
        bend_values = [p['Bend'][1] for p in params['individual_params']]
        
        # Write statistics
        f.write(f"CF: mean = {np.mean(cf_values):.4f}, std = {np.std(cf_values):.4f}\n")
        f.write(f"Spectral Slope: mean = {np.mean(slope_values):.6f}, std = {np.std(slope_values):.6f}\n")
        f.write(f"Restrahlen Stretch: mean = {np.mean(stretch_values):.4f}, std = {np.std(stretch_values):.4f}\n")
        f.write(f"Restrahlen Bend: mean = {np.mean(bend_values):.4f}, std = {np.std(bend_values):.4f}\n")

plt.show()