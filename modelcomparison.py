import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# Create save directory for comparison results
save_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters/test_comparison'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def extract_pvalues(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
        if 'knn_analysis_results.txt' in filepath:
            # Extract p-values from binomial tests
            pvalues = {}
            matches = re.findall(r'(\w+):\n.*\n.*\n.*\n.*p-value: (\d+\.\d+e?[-+]?\d*)', content)
            return {site: float(p_value) for site, p_value in matches}
            
        elif 'pca_analysis_results.txt' in filepath:
            # Extract p-values from feature-wise t-tests
            pvalues = {}
            sections = content.split('Feature-wise t-tests:')
            for section in sections[1:]:
                site_pair = re.search(r'(\w+) vs (\w+):', section)
                if site_pair:
                    s1, s2 = site_pair.groups()
                    matches = re.findall(r'\s+\w+: t = [-\d.]+, p = (\d+\.\d+e?[-+]?\d*)', section)
                    if matches:
                        pvalues[f"{s1} vs {s2}"] = min(float(p) for p in matches)
            return pvalues
            
        elif 'anova_analysis_results.txt' in filepath:
            # Extract p-values from Tukey's test
            pvalues = {}
            matches = re.findall(r'(\w+) vs (\w+):\n.*\n.*\n.*p-value: (\d+\.\d+e?[-+]?\d*)', content)
            return {f"{s1} vs {s2}": float(p_value) for s1, s2, p_value in matches}
            
        elif 'ratio_analysis_results.txt' in filepath:
            # Updated pattern to match all site comparisons in the ratio analysis
            pvalues = {}
            site_comparisons = re.findall(r'(\w+) vs (\w+):\nCentroid Distance:.*?\n((?:\s+BP\d/BP\d: t = [-\d.]+, p = \d+\.\d+e?[-+]?\d*\n\s+Significance:.*?\n)+)', content, re.DOTALL)
            
            for s1, s2, ratio_tests in site_comparisons:
                # Extract all p-values for this site pair
                p_values = re.findall(r'p = (\d+\.\d+e?[-+]?\d*)', ratio_tests)
                if p_values:
                    # Convert to floats and take minimum p-value
                    p_values = [float(p) for p in p_values]
                    pvalues[f"{s1} vs {s2}"] = min(p_values)
            
            return pvalues
    
    return {}

# Read results from each analysis
results_dir = '/Users/emmabelhadfa/Documents/Oxford/bandparameters'
test_results = {
    'ANOVA': extract_pvalues(os.path.join(results_dir, 'anova/anova_analysis_results.txt')),
    'KNN': extract_pvalues(os.path.join(results_dir, 'knearest/knn_analysis_results.txt')),
    'PCA': extract_pvalues(os.path.join(results_dir, 'pca/pca_analysis_results.txt')),
    'Band Ratios': extract_pvalues(os.path.join(results_dir, 'ratios/ratio_analysis_results.txt'))
}

# Get unique sites
sites = ['kingfisher', 'nightingale', 'osprey', 'sandpiper']

# Create comparison matrix between sites
n_sites = len(sites)
comparison_matrix = np.zeros((n_sites, n_sites, 4))  # 4 tests

# Fill comparison matrix with -log10(p) values
for i, site1 in enumerate(sites):
    for j, site2 in enumerate(sites):
        if i != j:
            site_pair = f"{site1} vs {site2}"
            rev_site_pair = f"{site2} vs {site1}"
            
            for k, (test_name, results) in enumerate(test_results.items()):
                p_value = None
                if site_pair in results:
                    p_value = results[site_pair]
                elif rev_site_pair in results:
                    p_value = results[rev_site_pair]
                elif test_name == 'KNN':
                    # For KNN, use the minimum p-value of the two sites
                    if site1 in results and site2 in results:
                        p_value = min(results[site1], results[site2])
                
                if p_value is not None:
                    comparison_matrix[i,j,k] = -np.log10(p_value)  # Convert to -log10(p) for better visualization

# Create visualization of test comparisons
plt.figure(figsize=(15, 10))

# 2x2 subplot for each statistical test
test_names = ['ANOVA', 'KNN', 'PCA', 'Band Ratios']
for i, test in enumerate(test_names):
    plt.subplot(2, 2, i+1)
    
    # Create matrix of -log10(p) values
    p_matrix = np.zeros((n_sites, n_sites))
    for i_site, site1 in enumerate(sites):
        for j_site, site2 in enumerate(sites):
            if i_site != j_site:
                site_pair = f"{site1} vs {site2}"
                rev_site_pair = f"{site2} vs {site1}"
                
                if site_pair in test_results[test]:
                    p_matrix[i_site, j_site] = -np.log10(test_results[test][site_pair])
                elif rev_site_pair in test_results[test]:
                    p_matrix[i_site, j_site] = -np.log10(test_results[test][rev_site_pair])
                elif test == 'KNN' and site1 in test_results[test] and site2 in test_results[test]:
                    p_matrix[i_site, j_site] = -np.log10(min(test_results[test][site1], test_results[test][site2]))
    
    im = sns.heatmap(p_matrix,
                     xticklabels=sites,
                     yticklabels=sites,
                     cmap='YlOrRd',
                     annot=True,
                     fmt='.2f')
    
    plt.title(f'{test}')
    cbar = im.collections[0].colorbar
    cbar.set_label('-log10(p)')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'test_comparison_heatmap.png'))
plt.show()

# Save numerical results
with open(os.path.join(save_dir, 'statistical_test_comparison.txt'), 'w') as f:
    f.write('Statistical Test Comparison Results\n')
    f.write('=================================\n\n')
    
    for test in test_names:
        f.write(f'\n{test} Results:\n')
        f.write('-' * (len(test) + 9) + '\n')
        
        if test in test_results:
            results = test_results[test]
            for key, value in results.items():
                f.write(f'{key}: p = {value:.2e}\n')
                
                # Add interpretation of significance
                if value < 0.001:
                    significance = "Highly significant"
                elif value < 0.01:
                    significance = "Very significant"
                elif value < 0.05:
                    significance = "Significant"
                else:
                    significance = "Not significant"
                f.write(f'Significance: {significance} (α = 0.05)\n\n')

# Create overall comparison plot
plt.figure(figsize=(12, 8))

# Create a matrix for the overall comparison
overall_comparison = np.zeros((n_sites, n_sites))

for i, site1 in enumerate(sites):
    for j, site2 in enumerate(sites):
        if i != j:
            p_values = []
            for test_name, results in test_results.items():
                site_pair = f"{site1} vs {site2}"
                rev_site_pair = f"{site2} vs {site1}"
                
                if site_pair in results:
                    p_values.append(results[site_pair])
                elif rev_site_pair in results:
                    p_values.append(results[rev_site_pair])
                elif test_name == 'KNN' and site1 in results and site2 in results:
                    p_values.append(min(results[site1], results[site2]))
            
            if p_values:
                overall_comparison[i,j] = -np.log10(min(p_values))

plt.subplot(1, 2, 1)
im = sns.heatmap(overall_comparison,
                 xticklabels=sites,
                 yticklabels=sites,
                 cmap='YlOrRd',
                 annot=True,
                 fmt='.2f')
plt.title('Overall Site Differences')
cbar = im.collections[0].colorbar
cbar.set_label('-log10(p)')

# Create bar plot of average significance by site
avg_significance = np.zeros(n_sites)
for i, site in enumerate(sites):
    site_p_values = []
    for test_name, results in test_results.items():
        if test_name == 'KNN':
            if site in results:
                site_p_values.append(results[site])
        else:
            for key, value in results.items():
                if site in key:
                    site_p_values.append(value)
    
    if site_p_values:
        avg_significance[i] = np.mean(site_p_values)

plt.subplot(1, 2, 2)
plt.bar(sites, avg_significance)
plt.xticks(rotation=45)
plt.ylabel('Average p-value')
plt.title('Average Site Significance\nAcross All Tests')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'overall_comparison.png'))
plt.show()
