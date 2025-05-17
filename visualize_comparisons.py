import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn as sns
import numpy as np
import os
from typing import cast

def load_comparison_data(comparison_file: str):
    """Load comparison data from CSV file"""
    if not os.path.exists(comparison_file):
        raise FileNotFoundError(f"Comparison file not found: {comparison_file}")
    return pd.read_csv(comparison_file)

def plot_performance_by_dataset(df: pd.DataFrame, output_dir: str='results/plots'):
    """Create bar plots of performance by dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Split by metric
    for metric_name, metric_group in df.groupby('metric'):
        metric_name = str(metric_name)
        nice_metric_name = 'Accuracy' if 'accuracy' in metric_name.lower() else 'ROC AUC'
        
        # Sort by user performance
        metric_group = metric_group.sort_values('user_value', ascending=False)
        
        plt.figure(figsize=(14, 10))
        
        # Dataset names can be long, so truncate them
        metric_group['short_name'] = metric_group['dataset_name'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
        
        # Create plot
        ax = plt.subplot(111)
        width = 0.35
        x = np.arange(len(metric_group))
        
        # Plot bars
        ax.bar(x - width/2, metric_group['user_value'], width, label='TabPFN', color='#1f77b4')
        ax.bar(x + width/2, metric_group['openml_mean'], width, label='OpenML Avg', color='#ff7f0e')
        
        # Add error bars for OpenML
        ax.errorbar(x + width/2, metric_group['openml_mean'], 
                   yerr=metric_group['openml_std'], fmt='none', color='k', capsize=3)
        
        # Mark best performance
        ax.scatter(x + width/2, metric_group['openml_max'], marker=MarkerStyle('*'), color='red', 
                  s=100, label='OpenML Best')
        
        # Configure plot
        ax.set_ylabel(nice_metric_name, fontsize=12)
        ax.set_title(f'TabPFN vs OpenML - {nice_metric_name} by Dataset', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_group['short_name'], rotation=90)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limits based on the metric
        if 'accuracy' in metric_name.lower():
            y_min = max(0, min(metric_group['user_value'].min(), 
                             metric_group['openml_mean'].min() - metric_group['openml_std'].max()) - 0.05)
            plt.ylim(y_min, 1.05)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{nice_metric_name.lower().replace(' ', '_')}_by_dataset.png", dpi=300)
        plt.close()

def plot_percentile_distribution(df: pd.DataFrame, output_dir: str='results/plots'):
    """Create histogram of percentile distribution"""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, metric_group in df.groupby('metric'):
        metric_name = str(metric_name)
        nice_metric_name = 'Accuracy' if 'accuracy' in metric_name.lower() else 'ROC AUC'
        
        plt.figure(figsize=(10, 6))
        sns.histplot(metric_group['percentile'], bins=20, kde=True)
        plt.axvline(x=50, color='r', linestyle='--', label='Median Performance')
        plt.axvline(x=metric_group['percentile'].mean(), color='g', linestyle='--', 
                   label=f'TabPFN Mean: {metric_group["percentile"].mean():.1f}th percentile')
        
        plt.xlabel('Percentile of Performance', fontsize=12)
        plt.ylabel('Count of Datasets', fontsize=12)
        plt.title(f'Distribution of TabPFN Performance Percentiles - {nice_metric_name}', fontsize=14)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/{nice_metric_name.lower().replace(' ', '_')}_percentile_distribution.png", dpi=300)
        plt.close()

def plot_performance_scatter(df: pd.DataFrame, output_dir: str='results/plots'):
    """Create scatter plot comparing TabPFN vs OpenML average"""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, metric_group in df.groupby('metric'):
        metric_name = str(metric_name)
        nice_metric_name = 'Accuracy' if 'accuracy' in metric_name.lower() else 'ROC AUC'
        
        plt.figure(figsize=(10, 8))
        
        # Plot scatter
        plt.scatter(metric_group['openml_mean'], metric_group['user_value'], 
                   alpha=0.7, s=50)
        
        # Add dataset labels for outliers
        for _, row in metric_group.iterrows():
            row: pd.Series
            z_score = float(cast(float, row['z_score']))
            percentile = float(cast(float, row['percentile']))
            openml_mean = float(cast(float, row['openml_mean']))
            user_value = float(cast(float, row['user_value']))
            dataset_name = str(row['dataset_name'])[:15]

            if abs(z_score) > 2 or percentile > 95 or percentile < 5:
                plt.annotate(dataset_name, 
                            (openml_mean, user_value),
                            fontsize=8)
        
        # Plot diagonal line (where TabPFN = OpenML average)
        min_val = min(metric_group['openml_mean'].min(), metric_group['user_value'].min())
        max_val = max(metric_group['openml_mean'].max(), metric_group['user_value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
                label='Equal Performance')
        
        plt.xlabel(f'OpenML Average {nice_metric_name}', fontsize=12)
        plt.ylabel(f'TabPFN {nice_metric_name}', fontsize=12)
        plt.title(f'TabPFN vs OpenML Average - {nice_metric_name}', fontsize=14)
        
        # Add counts of points above/below the line
        above = (metric_group['user_value'] > metric_group['openml_mean']).sum()
        below = (metric_group['user_value'] <= metric_group['openml_mean']).sum()
        plt.annotate(f'TabPFN better: {above} datasets', (0.05, 0.95), 
                    xycoords='axes fraction', fontsize=10)
        plt.annotate(f'OpenML better: {below} datasets', (0.05, 0.90), 
                    xycoords='axes fraction', fontsize=10)
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/{nice_metric_name.lower().replace(' ', '_')}_scatter.png", dpi=300)
        plt.close()

def plot_z_score_distribution(df: pd.DataFrame, output_dir: str='results/plots'):
    """Create box plot of z-scores by dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, metric_group in df.groupby('metric'):
        metric_name = str(metric_name)
        nice_metric_name = 'Accuracy' if 'accuracy' in metric_name.lower() else 'ROC AUC'
        
        # Sort by z-score
        metric_group = metric_group.sort_values('z_score', ascending=False)
        
        plt.figure(figsize=(12, 8))
        
        # Plot horizontal bar chart of z-scores
        ax = plt.subplot(111)
        
        # Color bars based on z-score
        colors = ['green' if z > 0 else 'red' for z in metric_group['z_score']]
        
        # Create plot
        ax.barh(metric_group['dataset_name'].str[:20], metric_group['z_score'], color=colors)
        
        # Configure plot
        ax.set_xlabel('Z-Score (Standard Deviations from Mean)', fontsize=12)
        ax.set_title(f'TabPFN Performance Relative to OpenML Average - {nice_metric_name}', fontsize=14)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{nice_metric_name.lower().replace(' ', '_')}_z_scores.png", dpi=300)
        plt.close()

def main():
    # Load comparison data
    comparison_file = 'results/tabpfn_vs_openml_comparison.csv'
    try:
        df = load_comparison_data(comparison_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the second script to generate the comparison data first.")
        return
    
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all plots
    print("Creating performance by dataset plots...")
    plot_performance_by_dataset(df, output_dir)
    
    print("Creating percentile distribution plots...")
    plot_percentile_distribution(df, output_dir)
    
    print("Creating performance scatter plots...")
    plot_performance_scatter(df, output_dir)
    
    print("Creating z-score distribution plots...")
    plot_z_score_distribution(df, output_dir)
    
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    main()
