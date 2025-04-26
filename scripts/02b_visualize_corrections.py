# scripts/02b_visualize_corrections.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_correction_results(results_file, quality_file, output_dir='plots/corrections'):
    """
    Create visualizations of correction results.
    
    Args:
        results_file: Path to correction results CSV
        quality_file: Path to original quality assessment CSV
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    results_df = pd.read_csv(results_file)
    quality_df = pd.read_csv(quality_file)
    
    # 1. Plot quality tier changes
    if 'original_quality_tier' in results_df.columns and 'new_quality_tier' in results_df.columns:
        # Create contingency table
        tier_changes = pd.crosstab(
            results_df['original_quality_tier'],
            results_df['new_quality_tier'],
            margins=True,
            margins_name='Total'
        )
        
        # Plot as heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(tier_changes.iloc[:-1, :-1], annot=True, fmt='d', cmap='viridis')
        plt.title('Quality Tier Changes After Correction')
        plt.xlabel('New Quality Tier')
        plt.ylabel('Original Quality Tier')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_tier_changes.png'), dpi=300)
        plt.close()
    
    # 2. Bar chart of correction methods
    if 'correction_methods' in results_df.columns:
        # Flatten the list of methods
        all_methods = []
        for methods in results_df['correction_methods']:
            if isinstance(methods, list):
                all_methods.extend(methods)
            elif isinstance(methods, str):
                # Handle string representation of list
                methods = methods.strip('[]').replace("'", "").split(', ')
                all_methods.extend([m for m in methods if m])
        
        # Count methods
        method_counts = pd.Series(all_methods).value_counts()
        
        plt.figure(figsize=(12, 6))
        method_counts.plot(kind='bar')
        plt.title('Correction Methods Applied')
        plt.xlabel('Method')
        plt.ylabel('Number of Stations')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correction_methods.png'), dpi=300)
        plt.close()
    
    # 3. Before-After comparison for key metrics
    metrics = ['sl_p99', 'sl_pct_missing', 'joint_exceedances']
    metric_labels = ['99th Percentile Sea Level (m)', 'Missing Sea Level Data (%)', 'Joint Exceedances Count']
    
    # Merge original quality with results
    merged_df = pd.merge(
        quality_df[['site_code'] + metrics],
        results_df[results_df['correction_status'] == 'success']['site_code'],
        on='site_code'
    )
    
    if not merged_df.empty:
        # Plot each metric
        for metric, label in zip(metrics, metric_labels):
            plt.figure(figsize=(12, 6))
            
            # Find corrected values
            corrected_file_dir = os.path.join('data', 'processed', 'corrected')
            
            # Get a few examples to show detailed before/after
            sample_stations = merged_df.sample(min(5, len(merged_df)))
            
            for i, (_, row) in enumerate(sample_stations.iterrows()):
                site_code = row['site_code']
                orig_value = row[metric]
                
                # Load corrected data for this station
                corrected_file = os.path.join(corrected_file_dir, f"{site_code}_corrected.csv")
                
                if os.path.exists(corrected_file):
                    # Load corrected data
                    corr_df = pd.read_csv(corrected_file, parse_dates=['datetime'])
                    
                    # Calculate new metric value
                    if metric == 'sl_p99':
                        valid_sl = corr_df[corr_df['sea_level'] != -99.9999]['sea_level']
                        new_value = valid_sl.quantile(0.99) if not valid_sl.empty else np.nan
                    elif metric == 'sl_pct_missing':
                        new_value = 100 * (corr_df['sea_level'] == -99.9999).sum() / len(corr_df)
                    elif metric == 'joint_exceedances':
                        # Need to recalculate joint exceedances
                        valid_sl = corr_df[corr_df['sea_level'] != -99.9999]['sea_level']
                        sl_threshold = valid_sl.quantile(0.95) if not valid_sl.empty else np.nan
                        
                        if 'precipitation_mm' in corr_df.columns and not np.isnan(sl_threshold):
                            precip_threshold = corr_df['precipitation_mm'].quantile(0.95)
                            sl_exceed = (corr_df['sea_level'] != -99.9999) & (corr_df['sea_level'] > sl_threshold)
                            precip_exceed = corr_df['precipitation_mm'] > precip_threshold
                            new_value = (sl_exceed & precip_exceed).sum()
                        else:
                            new_value = np.nan
                    
                    # Plot before/after
                    plt.subplot(1, len(sample_stations), i+1)
                    plt.bar(['Before', 'After'], [orig_value, new_value])
                    plt.title(f"Station {site_code}")
                    plt.xticks(rotation=0)
                    
                    # Add percent change
                    if not np.isnan(new_value) and not np.isnan(orig_value) and orig_value != 0:
                        pct_change = (new_value - orig_value) / orig_value * 100
                        plt.text(1, max(orig_value, new_value), f"{pct_change:.1f}%", 
                                ha='center', va='bottom')
            
            plt.suptitle(f'Before-After Comparison: {label}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'before_after_{metric}.png'), dpi=300)
            plt.close()

if __name__ == "__main__":
    results_file = os.path.join('data', 'processed', 'correction_results.csv')
    quality_file = os.path.join('data', 'processed', 'station_quality_assessment.csv')
    
    plot_correction_results(results_file, quality_file)
    print("Correction visualization completed. Plots saved to plots/corrections/")