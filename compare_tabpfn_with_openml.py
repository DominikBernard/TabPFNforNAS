import json
import pandas as pd
import os
import pickle
from scipy import stats

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

def compute_performance_metrics(user_results: pd.DataFrame, openml_evaluations: dict):
    """Compute comparison metrics between user results and OpenML evaluations"""
    comparison_rows = []
    
    metrics_map = {
        'predictive_accuracy': 'accuracy',
        'area_under_roc_curve': 'roc_auc'
    }
    
    for _, row in user_results.iterrows():
        task_id = row['task_id']
        
        if task_id in openml_evaluations:
            task_data = openml_evaluations[task_id]
            dataset_name = task_data.get('dataset_name', f"Unknown-{task_id}")
            
            for openml_metric, user_metric in metrics_map.items():
                if openml_metric in task_data and user_metric in row:
                    metric_data = task_data[openml_metric]
                    user_value = row[user_metric]
                    
                    # Calculate percentile and z-score
                    mean = metric_data['mean']
                    std = metric_data['std'] if metric_data['std'] > 0 else 1e-10  # Avoid division by zero
                    count = metric_data['count']
                    max_val = metric_data['max']
                    
                    z_score = (user_value - mean) / std
                    # Approximate percentile using z-score
                    percentile = stats.norm.cdf(z_score) * 100
                    
                    # Find rank (approximately)
                    rank = int(count * (1 - percentile / 100)) + 1
                    
                    # Find best flow
                    best_flow = None
                    if 'top_flows' in metric_data and len(metric_data['top_flows']) > 0:
                        best_flow = metric_data['top_flows'][0]
                    
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
                        'z_score': z_score,
                        'percentile': percentile,
                        'approximate_rank': rank,
                        'best_flow_id': best_flow['flow_id'] if best_flow else None,
                        'best_flow_name': best_flow['flow_name'] if best_flow else None,
                        'best_flow_value': best_flow['value'] if best_flow else None,
                        'difference_from_best': best_flow['value'] - user_value if best_flow else 0
                    }
                    
                    comparison_rows.append(comparison_row)
    
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
    
    top_percentile = (comparison_df['percentile'] >= 90).sum()
    print(f"Tasks where TabPFN is in the top 10% performers: {top_percentile}/{total} ({top_percentile/total*100:.1f}%)")
    
    beats_best = (comparison_df['difference_from_best'] >= 0).sum()
    print(f"Tasks where TabPFN matches or beats the best flow: {beats_best}/{total} ({beats_best/total*100:.1f}%)")
    
    # Performance by metric
    print("\nPerformance by Metric:")
    metric_summary = comparison_df.groupby('metric').agg({
        'z_score': ['mean', 'min', 'max'],
        'percentile': ['mean', 'min', 'max'],
        'task_id': 'count'
    })
    print(metric_summary)
    
    # Top and bottom performances
    print("\nTop 5 Performances (highest percentile):")
    top_5 = comparison_df.sort_values('percentile', ascending=False).head(5)
    print(top_5[['dataset_name', 'metric', 'percentile', 'user_value', 'openml_mean']])
    
    print("\nBottom 5 Performances (lowest percentile):")
    bottom_5 = comparison_df.sort_values('percentile').head(5)
    print(bottom_5[['dataset_name', 'metric', 'percentile', 'user_value', 'openml_mean']])

if __name__ == "__main__":
    main()
