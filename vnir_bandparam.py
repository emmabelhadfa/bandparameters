import os
import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.signal import savgol_filter

def combine_and_plot_ovirs_spectra(folder_paths):
    """
    Combines and averages OVIRS spectra from .fits files across multiple folders.
    Processes specific dates and combines related sites:
    Nightingale (20191026, 20200122)
    Kingfisher (20191019)
    Osprey (20191012, 20200211)
    Sandpiper (20191005)
    
    Parameters:
        folder_paths (list): List of paths to folders containing .fits and .xml files
    """
    # Define allowed prefixes with their site names
    site_mappings = {
        '20191026': 'Nightingale',
        '20200122': 'Nightingale',  # Additional Nightingale data
        '20191019': 'Kingfisher',
        '20191012': 'Osprey',
        '20200211': 'Osprey',      # Additional Osprey data
        '20191005': 'Sandpiper'
    }
    
    # Initialize dictionaries to store data by site
    site_data = {
        'Nightingale': {'waves': [], 'specs': []},
        'Kingfisher': {'waves': [], 'specs': []},
        'Osprey': {'waves': [], 'specs': []},
        'Sandpiper': {'waves': [], 'specs': []}
    }
    
    # Process each folder
    for folder_path in folder_paths:
        # Get all .fits files in folder that contain 'sci'
        fits_files = [f for f in os.listdir(folder_path) 
                     if f.endswith('.fits') and 'sci' in f.lower()]
        
        # Process each file
        for fits_file in fits_files:
            prefix = fits_file[:8]
            if prefix in site_mappings:
                site_name = site_mappings[prefix]
                
                # Get corresponding xml filename
                xml_file = fits_file.replace('.fits', '.xml')
                
                # Full paths
                fits_path = os.path.join(folder_path, fits_file)
                xml_path = os.path.join(folder_path, xml_file)
                
                # Skip if xml doesn't exist
                if not os.path.exists(xml_path):
                    print(f"Warning: No XML file found for {fits_file}")
                    continue
                
                # Read FITS file
                with fits.open(fits_path) as hdul:
                    spectral_data = hdul[0].data  # 2D spectrum array
                    wavelengths = hdul[1].data    # 1D wavelength array in HDU 1
                    
                    # Get spectral data for the first line
                    spec = spectral_data[0, :]    # Shape is (1, 1393)
                    wave = wavelengths            # Shape is (1393,)
                    
                    site_data[site_name]['waves'].append(wave)
                    site_data[site_name]['specs'].append(spec)
    
    # Store normalized data for final comparison plot
    comparison_data = {}
    
    # Process each site's data
    for site_name, data in site_data.items():
        if data['waves']:  # Only process if we have data for this site
            print(f"\nProcessing {site_name}")
            print(f"Number of spectra: {len(data['waves'])}")
            
            # Convert to arrays
            all_waves = np.array(data['waves'])
            all_specs = np.array(data['specs'])
            
            # Calculate SNR
            signal_mean = np.mean(all_specs, axis=0)
            signal_std = np.std(all_specs, axis=0)
            snr = np.abs(signal_mean / (signal_std + 1e-10))  # Add small number to avoid division by zero
            median_snr = np.median(snr)
            print(f"Median SNR: {median_snr:.2f}")
            
            # Average the spectra
            avg_wave = np.mean(all_waves, axis=0)
            avg_spec = np.mean(all_specs, axis=0)
            
            # Filter wavelength range to 0.4-3.8 μm
            wave_mask = (avg_wave >= 0.4) & (avg_wave <= 3.8)
            avg_wave = avg_wave[wave_mask]
            avg_spec = avg_spec[wave_mask]
            
            # Apply Savitzky-Golay smoothing
            # window_length must be odd and less than data points
            window_length = 11  # Adjust this for more/less smoothing
            polyorder = 3      # Adjust this to change the polynomial order
            spec_smooth = savgol_filter(avg_spec, window_length, polyorder)
            
            # Find maximum value for normalization
            max_idx = np.argmax(spec_smooth)
            print(f"Normalization wavelength (max value): {avg_wave[max_idx]:.3f} μm")
            
            # Normalize both raw and smoothed data
            spec_normalized = avg_spec / avg_spec[max_idx]
            spec_smooth_normalized = spec_smooth / spec_smooth[max_idx]
            
            # Store the normalized smoothed data for comparison plot
            comparison_data[site_name] = {
                'wavelength': avg_wave,
                'spectrum': spec_smooth_normalized
            }
            
            # Create individual site plots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot 1: Raw averaged data with smoothed overlay
            ax1.plot(avg_wave, avg_spec, 'b-', alpha=0.5, label='Raw Average Spectrum')
            ax1.plot(avg_wave, spec_smooth, 'r-', label='Smoothed Spectrum')
            ax1.set_title(f'Raw and Smoothed Average OVIRS Spectrum - {site_name}')
            ax1.set_xlabel('Wavelength (μm)')
            ax1.set_ylabel('Reflectance')
            ax1.set_xlim(0.4, 3.8)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
            
            # Add wavenumber axis to plot 1
            ax1_top = ax1.twiny()
            ax1_top.set_xlim(10000/3.8, 10000/0.4)  # Convert wavelength to wavenumber
            ax1_top.set_xlabel('Wavenumber (cm⁻¹)')
            
            # Plot 2: Normalized data
            ax2.plot(avg_wave, spec_normalized, 'b-', alpha=0.5, label='Raw Normalized')
            ax2.plot(avg_wave, spec_smooth_normalized, 'r-', label='Smoothed Normalized')
            ax2.plot(avg_wave[max_idx], spec_smooth_normalized[max_idx], 'k.',
                    label=f'Normalization point')
            ax2.set_title(f'OVIRS Normalized Spectrum - {site_name}')
            ax2.set_xlabel('Wavelength (μm)')
            ax2.set_ylabel('Normalized Reflectance')
            ax2.set_xlim(0.4, 3.8)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
            
            # Add wavenumber axis to plot 2
            ax2_top = ax2.twiny()
            ax2_top.set_xlim(10000/3.8, 10000/0.4)
            ax2_top.set_xlabel('Wavenumber (cm⁻¹)')
            
            # Plot 3: Signal-to-Noise Ratio
            ax3.semilogy(avg_wave, snr[wave_mask], 'g-', label='SNR')
            ax3.axhline(y=median_snr, color='r', linestyle='--', 
                       label=f'Median SNR: {median_snr:.2f}')
            ax3.set_title(f'Signal-to-Noise Ratio - {site_name}')
            ax3.set_xlabel('Wavelength (μm)')
            ax3.set_ylabel('SNR (log scale)')
            ax3.set_xlim(0.4, 3.8)
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
            
            # Add wavenumber axis to plot 3
            ax3_top = ax3.twiny()
            ax3_top.set_xlim(10000/3.8, 10000/0.4)
            ax3_top.set_xlabel('Wavenumber (cm⁻¹)')
            
            plt.tight_layout()
            plt.show()

    # Create comparison plots
    if comparison_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['b', 'g', 'r', 'purple']
        
        # Create save directory if it doesn't exist
        save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters/vnir'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for (site_name, data), color in zip(comparison_data.items(), colors):
            # Plot the spectrum
            ax.plot(data['wavelength'], data['spectrum'], color=color, label=f"{site_name}")
            
            # Calculate parameters
            params = calculate_spectral_parameters(data['wavelength'], data['spectrum'])
            
            # Save parameters to text file
            output_file = os.path.join(save_dir, f'{site_name}_spectral_parameters.txt')
            with open(output_file, 'w') as f:
                f.write(f"Spectral Parameters for {site_name}\n")
                f.write("-" * 50 + "\n\n")
                
                # MIN2295_2480
                f.write("MIN2295_2480:\n")
                f.write(f"Value: {params['MIN2295_2480']:.6f}\n")
                f.write("Component reflectance values:\n")
                f.write(f"  R(2.295): {params['reflectance_values']['2.295']:.6f}\n")
                f.write(f"  R(2.165): {params['reflectance_values']['2.165']:.6f}\n")
                f.write(f"  R(2.364): {params['reflectance_values']['2.364']:.6f}\n")
                f.write(f"  R(2.480): {params['reflectance_values']['2.480']:.6f}\n")
                f.write(f"  R(2.570): {params['reflectance_values']['2.570']:.6f}\n\n")
                
                # MIN2345_2537
                f.write("MIN2345_2537:\n")
                f.write(f"Value: {params['MIN2345_2537']:.6f}\n")
                f.write("Component reflectance values:\n")
                f.write(f"  R(2.345): {params['reflectance_values']['2.345']:.6f}\n")
                f.write(f"  R(2.250): {params['reflectance_values']['2.250']:.6f}\n")
                f.write(f"  R(2.430): {params['reflectance_values']['2.430']:.6f}\n")
                f.write(f"  R(2.537): {params['reflectance_values']['2.537']:.6f}\n")
                f.write(f"  R(2.602): {params['reflectance_values']['2.602']:.6f}\n\n")
                
                # BD2500
                f.write("BD2500:\n")
                f.write(f"Value: {params['BD2500']:.6f}\n")
                f.write("Component reflectance values:\n")
                f.write(f"  R(2.480): {params['reflectance_values']['2.480']:.6f}\n")
                f.write(f"  R(2.364): {params['reflectance_values']['2.364']:.6f}\n")
                f.write(f"  R(2.570): {params['reflectance_values']['2.570']:.6f}\n\n")
                
                # BD3400
                f.write("BD3400:\n")
                f.write(f"Value: {params['BD3400']:.6f}\n")
                f.write("Component reflectance values:\n")
                f.write(f"  R(3.420): {params['reflectance_values']['3.420']:.6f}\n")
                f.write(f"  R(3.250): {params['reflectance_values']['3.250']:.6f}\n")
                f.write(f"  R(3.630): {params['reflectance_values']['3.630']:.6f}\n\n")
                
                # ICER
                f.write("ICER:\n")
                f.write(f"Value (slope): {params['ICER']:.6f}\n")
                f.write("Component reflectance values:\n")
                f.write(f"  R(2.456): {params['reflectance_values']['2.456']:.6f}\n")
                f.write(f"  R(2.530): {params['reflectance_values']['2.530']:.6f}\n")
            
            # Plot markers for each parameter
            # MIN2295_2480
            wave_2295 = 2.295
            wave_2480 = 2.480
            idx_2295 = np.argmin(np.abs(data['wavelength'] - wave_2295))
            idx_2480 = np.argmin(np.abs(data['wavelength'] - wave_2480))
            ax.plot(wave_2295, data['spectrum'][idx_2295], 'ro', 
                   label='MIN2295_2480' if site_name == list(comparison_data.keys())[0] else '')
            ax.plot(wave_2480, data['spectrum'][idx_2480], 'ro')
            
            # MIN2345_2537
            wave_2345 = 2.345
            wave_2537 = 2.537
            idx_2345 = np.argmin(np.abs(data['wavelength'] - wave_2345))
            idx_2537 = np.argmin(np.abs(data['wavelength'] - wave_2537))
            ax.plot(wave_2345, data['spectrum'][idx_2345], 'go',
                   label='MIN2345_2537' if site_name == list(comparison_data.keys())[0] else '')
            ax.plot(wave_2537, data['spectrum'][idx_2537], 'go')
            
            # BD2500
            wave_2500 = 2.480  # Using actual wavelength from formula
            idx_2500 = np.argmin(np.abs(data['wavelength'] - wave_2500))
            ax.plot(wave_2500, data['spectrum'][idx_2500], 'bo',
                   label='BD2500' if site_name == list(comparison_data.keys())[0] else '')
            
            # BD3400
            wave_3400 = 3.420
            idx_3400 = np.argmin(np.abs(data['wavelength'] - wave_3400))
            ax.plot(wave_3400, data['spectrum'][idx_3400], 'mo',
                   label='BD3400' if site_name == list(comparison_data.keys())[0] else '')
            
            # ICER slope
            wave_2456 = 2.456
            wave_2530 = 2.530
            idx_2456 = np.argmin(np.abs(data['wavelength'] - wave_2456))
            idx_2530 = np.argmin(np.abs(data['wavelength'] - wave_2530))
            ax.plot([wave_2456, wave_2530], 
                   [data['spectrum'][idx_2456], data['spectrum'][idx_2530]], '--k',
                   label='ICER' if site_name == list(comparison_data.keys())[0] else '')
        
        ax.set_title('Comparison of Smoothed Normalized OVIRS Spectra Across Sites')
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Normalized Reflectance')
        ax.set_xlim(0.4, 3.8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Add wavenumber axis
        ax_top = ax.twiny()
        ax_top.set_xlim(10000/3.8, 10000/0.4)
        ax_top.set_xlabel('Wavenumber (cm⁻¹)')
        
        plt.tight_layout()
        plt.show()

def calculate_spectral_parameters(wavelength, spectrum):
    """Calculate spectral parameters for a given spectrum."""
    
    # Helper function to get reflectance at specific wavelength
    def get_reflectance(wave_target):
        # Find the index of the closest wavelength value
        idx = np.argmin(np.abs(wavelength - wave_target))
        # Return the reflectance (y-value) at that wavelength
        return spectrum[idx]
    
    # Get reflectance values for MIN2295_2480
    r2295 = get_reflectance(2.295)  # reflectance at 2.295 μm
    r2165 = get_reflectance(2.165)  # reflectance at 2.165 μm
    r2364 = get_reflectance(2.364)  # reflectance at 2.364 μm
    r2480 = get_reflectance(2.480)  # reflectance at 2.480 μm
    r2570 = get_reflectance(2.570)  # reflectance at 2.570 μm
    
    # Calculate MIN2295_2480
    min2295 = 1 - (r2295 / (r2165 + r2364))  # First minimum
    min2480 = 1 - (r2480 / (r2364 + r2570))  # Second minimum
    MIN2295_2480 = min(min2295, min2480)      # Take the smaller value
    
    # Get reflectance values for MIN2345_2537
    r2345 = get_reflectance(2.345)  # reflectance at 2.345 μm
    r2250 = get_reflectance(2.250)  # reflectance at 2.250 μm
    r2430 = get_reflectance(2.430)  # reflectance at 2.430 μm
    r2537 = get_reflectance(2.537)  # reflectance at 2.537 μm
    r2602 = get_reflectance(2.602)  # reflectance at 2.602 μm
    
    # Calculate MIN2345_2537
    min2345 = 1 - (r2345 / (r2250 + r2430))  # First minimum
    min2537 = 1 - (r2537 / (r2430 + r2602))  # Second minimum
    MIN2345_2537 = min(min2345, min2537)      # Take the smaller value
    
    # Calculate BD2500 (using same reflectance values as above)
    BD2500 = 1 - (r2480 / (r2364 + r2570))
    
    # Get reflectance values for BD3400
    r3420 = get_reflectance(3.420)  # reflectance at 3.420 μm
    r3250 = get_reflectance(3.250)  # reflectance at 3.250 μm
    r3630 = get_reflectance(3.630)  # reflectance at 3.630 μm
    
    # Calculate BD3400
    BD3400 = 1 - (r3420 / (r3250 + r3630))
    
    # Get reflectance values for ICER
    r2456 = get_reflectance(2.456)  # reflectance at 2.456 μm
    r2530 = get_reflectance(2.530)  # reflectance at 2.530 μm
    
    # Calculate ICER (slope)
    ICER = (r2530 - r2456) / (2.530 - 2.456)
    
    # Return all parameters and their component reflectance values
    return {
        'MIN2295_2480': MIN2295_2480,
        'MIN2345_2537': MIN2345_2537,
        'BD2500': BD2500,
        'BD3400': BD3400,
        'ICER': ICER,
        # Also store the individual reflectance values
        'reflectance_values': {
            '2.295': r2295,
            '2.165': r2165,
            '2.364': r2364,
            '2.480': r2480,
            '2.570': r2570,
            '2.345': r2345,
            '2.250': r2250,
            '2.430': r2430,
            '2.537': r2537,
            '2.602': r2602,
            '3.420': r3420,
            '3.250': r3250,
            '3.630': r3630,
            '2.456': r2456,
            '2.530': r2530
        }
    }

if __name__ == "__main__":
    # Example usage - replace with your folder paths
    folder_paths = [
        '/Users/emmabelhadfa/Documents/Oxford/OVIRS/spectraldata/ovirs/recon',
        '/Users/emmabelhadfa/Documents/Oxford/OVIRS/spectraldata/ovirs/recon_b'  # Add your second folder path here
    ]
    combine_and_plot_ovirs_spectra(folder_paths)
