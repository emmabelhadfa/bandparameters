import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing HDF5 files
data_directory = "/Users/emmabelhadfa/Documents/Oxford/orex/OTES/data/locations/"

# Get all HDF5 file paths in the directory
file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith(".hdf5")]

# Define band parameter regions
bp1_region = (950, 1150)  # BP1 region
bp2_region = (800, 950)   # BP2 region 
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
    non_outlier_emissivity = mt_emissivity_clipped[:, :, non_outliers[non_outliers < mt_emissivity_clipped.shape[2]]]
    site_average = np.mean(non_outlier_emissivity, axis=2)[:, 0]
    
    # Calculate band parameters
    bp_values = []
    for region in [bp1_region, bp2_region, bp4_region]:
        mask = (xaxis_L3 >= region[0]) & (xaxis_L3 <= region[1])
        bp_value = np.mean(site_average[mask])
        bp_values.append(bp_value)
    
    band_params_by_site[site_name] = bp_values

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Extract band parameters for plotting
sites = list(band_params_by_site.keys())
bp1_values = [band_params_by_site[site][0] for site in sites]
bp2_values = [band_params_by_site[site][1] for site in sites]
bp4_values = [band_params_by_site[site][2] for site in sites]

# Create scatter plot
scatter = ax.scatter(bp1_values, bp2_values, bp4_values, c='red', s=100)

# Add site labels to points
for i, site in enumerate(sites):
    ax.text(bp1_values[i], bp2_values[i], bp4_values[i], 
            site, fontsize=8)

ax.set_xlabel('BP1 (950-1150 cm⁻¹)')
ax.set_ylabel('BP2 (800-950 cm⁻¹)')
ax.set_zlabel('BP4 (400-500 cm⁻¹)')
ax.set_title('3D Visualization of Band Parameters')

# Adjust the view angle for better visualization
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# Print numerical results
print("\nBand Parameters by Site:")
for site in sites:
    print(f"\n{site}:")
    print(f"BP1 (950-1150 cm⁻¹): {band_params_by_site[site][0]:.3f}")
    print(f"BP2 (800-950 cm⁻¹): {band_params_by_site[site][1]:.3f}")
    print(f"BP4 (400-500 cm⁻¹): {band_params_by_site[site][2]:.3f}")