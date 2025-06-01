import json
import pandas as pd
import os
import pickle
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import cast
from tqdm import tqdm

def load_user_results(results_file: str):
    """Load user's benchmark results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame for easier processing
    results_df = pd.DataFrame(data['results'])
    
    # Map metrics to OpenML metric names
    results_df['predictive_accuracy'] = results_df['accuracy']
    results_df['area_under_roc_curve'] = results_df['roc_auc']
    
    return results_df

def load_openml_evaluations(cache_file: str):
    """Load OpenML evaluations from cache file"""
    with open(cache_file, 'rb') as f:
        return pickle.load(f)
    
def create_violin_plot(data: pd.DataFrame, user_result: pd.DataFrame, dataset_name: str, metric: str, overall: bool = False):
    """Create a violin plot for the given data and user result"""

    if overall:
        plt.figure(figsize=(50, 10))
        directory = 'results/plots/overall'
        plt.xticks(rotation=75)
    else:
        plt.figure(figsize=(8, 6))
        directory = 'results/plots'
    
    plt.tight_layout()
    ax = sns.violinplot(x=data['dataset_name'], y=data['value'], inner="point", split=True, density_norm='width', cut=0)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    # Add user result line
    for i, row in user_result.iterrows():
        x = labels.index(cast(str, row['dataset_name']))
        y = row['value']
        ax.hlines(y, x - 0.3, x + 0.3, color='red', linewidth=2, label='User Result' if i == 0 else "")
    
    plt.title(f"Violin Plot for {dataset_name} - {metric}")
    plt.xlabel('Dataset')
    plt.ylabel(metric)
    plt.legend()
    
    # Save the plot
    plt.savefig(f'{directory}/violin_plot_{dataset_name}_{metric}.png', bbox_inches='tight')
    plt.close()

def compute_performance_metrics(user_results: pd.DataFrame, openml_evaluations: dict):
    """Compute comparison metrics between user results and OpenML evaluations"""
    comparison_rows = []
    
    metrics_map = {
        'predictive_accuracy': 'accuracy',
        'area_under_roc_curve': 'roc_auc'
    }
    
    accuracy_user = pd.DataFrame(columns=['dataset_name', 'value'])
    roc_auc_user = pd.DataFrame(columns=['dataset_name', 'value'])
    accuracy_openml = pd.DataFrame(columns=['dataset_name', 'value'])
    roc_auc_openml = pd.DataFrame(columns=['dataset_name', 'value'])

    for _, row in tqdm(user_results.iterrows(), desc="Processing user results", total=len(user_results)):
        task_id = row['task_id']
        
        if task_id in openml_evaluations:
            task_data = openml_evaluations[task_id]
            dataset_name = task_data.get('dataset_name', f"Unknown-{task_id}")
            
            for openml_metric, user_metric in metrics_map.items():
                if openml_metric in task_data and user_metric in row:
                    metric_data = task_data[openml_metric]
                    user_value = float(cast(float, row[user_metric]))
                    
                    mean = metric_data['mean']
                    std = metric_data['std'] if metric_data['std'] > 0 else 1e-10  # Avoid division by zero
                    count = metric_data['count']
                    max_val = metric_data['max']

                    sorted_runs = pd.DataFrame(metric_data['sorted_runs'])
                    sorted_runs = sorted_runs[['data_name', 'value']].rename(columns={'data_name': 'dataset_name'})
                    
                    rank = stats.percentileofscore(sorted_runs['value'], user_value, kind='mean')
                                        
                    # Find best flow
                    best_flow = None
                    if 'sorted_runs' in metric_data and len(metric_data['sorted_runs']) > 0:
                        best_flow = metric_data['sorted_runs'][0]
                    
                    comparison_row = {
                        'task_id': task_id,
                        'dataset_id': row['dataset_id'],
                        'dataset_name': dataset_name,
                        'metric': openml_metric,
                        'user_value': user_value,
                        'openml_mean': mean,
                        'openml_std': std,
                        'openml_max': max_val,
                        'total_runs': count,
                        'rank': rank,
                        'best_run_id': best_flow['run_id'] if best_flow else None,
                        'best_run_name': best_flow['flow_name'] if best_flow else None,
                        'best_run_value': best_flow['value'] if best_flow else None,
                        'difference_from_best': best_flow['value'] - user_value if best_flow else 0
                    }
                    
                    comparison_rows.append(comparison_row)

                    user_value_df = pd.DataFrame({'dataset_name': [dataset_name], 'value': [user_value]})
                    # Collect values for overall statistics
                    if user_metric == 'accuracy':
                        accuracy_user = pd.concat([accuracy_user, user_value_df] if not accuracy_user.empty else [user_value_df], ignore_index=True)
                        accuracy_openml = pd.concat([accuracy_openml, sorted_runs] if not accuracy_openml.empty else [sorted_runs], ignore_index=True)
                    elif user_metric == 'roc_auc':
                        roc_auc_user = pd.concat([roc_auc_user, user_value_df] if not roc_auc_user.empty else [user_value_df], ignore_index=True)
                        roc_auc_openml = pd.concat([roc_auc_openml, sorted_runs] if not roc_auc_openml.empty else [sorted_runs], ignore_index=True)

                    # create violin plot data
                    create_violin_plot(
                        sorted_runs,
                        user_value_df,
                        dataset_name,
                        user_metric
                    )

    # create final violin plots
    create_violin_plot(
        accuracy_openml,
        accuracy_user,
        'Overall Accuracy',
        'predictive_accuracy',
        overall=True
    )

    create_violin_plot(
        roc_auc_openml,
        roc_auc_user,
        'Overall ROC AUC',
        'area_under_roc_curve',
        overall=True
    )
    
    return pd.DataFrame(comparison_rows)

def main():
    # Load user results
    user_results_file = 'results/benchmark_99_20250516_203740_full_output.json'
    user_results = load_user_results(user_results_file)
    
    # Load OpenML evaluations
    openml_evaluations_file = 'results/openml_cc18_evaluations.pkl'
    if not os.path.exists(openml_evaluations_file):
        print(f"Error: OpenML evaluations file not found at {openml_evaluations_file}")
        print("Please run the first script to download the evaluations first.")
        return
    
    openml_evaluations = load_openml_evaluations(openml_evaluations_file)
    
    # Compute comparison metrics
    comparison_df = compute_performance_metrics(user_results, openml_evaluations)
    
    # Save results
    comparison_df.to_csv('results/tabpfn_vs_openml_comparison.csv', index=False)
    print(f"Saved comparison results to results/tabpfn_vs_openml_comparison.csv")
    
    # Print summary statistics
    print("\nSummary of TabPFN performance:")
    
    # Overall performance statistics
    print("\nOverall Performance Statistics:")
    above_avg = (comparison_df['user_value'] > comparison_df['openml_mean']).sum()
    total = len(comparison_df)
    print(f"Tasks where TabPFN outperforms average: {above_avg}/{total} ({above_avg/total*100:.1f}%)")
    
    top_percentile = (comparison_df['rank'] >= 90).sum()
    print(f"Tasks where TabPFN is in the top 10% performers: {top_percentile}/{total} ({top_percentile/total*100:.1f}%)")
    
    beats_best = (comparison_df['difference_from_best'] <= 0).sum()
    print(f"Tasks where TabPFN matches or beats the best run: {beats_best}/{total} ({beats_best/total*100:.1f}%)")
    
    # Performance by metric
    print("\nPerformance by Metric:")
    metric_summary = comparison_df.groupby('metric').agg({
        'rank': ['mean', 'min', 'max'],
        'task_id': 'count'
    })
    print(metric_summary)
    
    # Top and bottom performances
    print("\nTop 5 Performances (highest rank):")
    top_5 = comparison_df.sort_values('rank', ascending=False).head(5)
    print(top_5[['dataset_name', 'metric', 'rank', 'user_value', 'openml_mean']])
    
    print("\nBottom 5 Performances (lowest rank):")
    bottom_5 = comparison_df.sort_values('rank').head(5)
    print(bottom_5[['dataset_name', 'metric', 'rank', 'user_value', 'openml_mean']])

if __name__ == "__main__":
    main()
