import openml
import pandas as pd
import os
from typing import List, Optional
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

def download_benchmark_evaluations(benchmark_id: int=99, metrics: Optional[List[str]]=None, cache_file: Optional[str]=None):
    """
    Download evaluation statistics for all tasks in a benchmark suite.
    
    Parameters:
    -----------
    benchmark_id : int
        ID of the OpenML benchmark suite
    metrics : list
        List of metrics to download (e.g., ['predictive_accuracy', 'area_under_roc_curve'])
    cache_file : str
        Path to cache file. If exists, will load from this file instead of downloading.
        
    Returns:
    --------
    dict: Dictionary with task_id as key and evaluation statistics as values
    """
    if metrics is None:
        metrics = ['predictive_accuracy', 'area_under_roc_curve']
    
    # Check if cache file exists and load it if it does
    if cache_file and os.path.exists(cache_file):
        print(f"Loading cached evaluations from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Get benchmark suite
    benchmark_suite = openml.study.get_suite(benchmark_id)
    print(f"Downloading evaluation data for benchmark: {benchmark_suite.name}")
    
    # Initialize results dictionary
    all_evaluations = {}
    
    # Download evaluations for each task and metric
    for task_id in tqdm(benchmark_suite.tasks, desc="Downloading tasks"):
        task_evaluations = {}
        
        for metric in metrics:
            try:
                # Get evaluations for this task and metric
                evals = openml.evaluations.list_evaluations(
                    function=metric,
                    tasks=[task_id],
                    size=10000,  # Limit to 10000 evaluations per task
                    output_format='dataframe'
                )
                
                if evals is not None and not evals.empty:
                    # Calculate statistics
                    mean_val = evals['value'].mean()
                    std_val = evals['value'].std()
                    max_val = evals['value'].max()
                    min_val = evals['value'].min()
                    
                    # Get top 5 flows
                    top_flows = evals.sort_values('value', ascending=False).head(5)
                    
                    # Store statistics
                    task_evaluations[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'max': max_val,
                        'min': min_val,
                        'count': len(evals),
                        'top_flows': top_flows[['flow_id', 'flow_name', 'value']].to_dict('records')
                    }
                    
                    print(f"Task {task_id}, Metric {metric}: {len(evals)} evaluations")
                else:
                    print(f"No evaluations found for task {task_id}, metric {metric}")
                    
            except Exception as e:
                print(f"Error downloading evaluations for task {task_id}, metric {metric}: {e}")
        
        if task_evaluations:
            # Get task and dataset info
            try:
                task = openml.tasks.get_task(task_id)
                dataset = task.get_dataset()
                task_evaluations['dataset_name'] = dataset.name
                task_evaluations['dataset_id'] = dataset.dataset_id
            except:
                task_evaluations['dataset_name'] = f"Unknown-{task_id}"
                task_evaluations['dataset_id'] = None
            
            all_evaluations[task_id] = task_evaluations
    
    # Save to cache file if provided
    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(all_evaluations, f)
        print(f"Cached evaluations to {cache_file}")
    
    return all_evaluations

if __name__ == "__main__":
    # Download evaluations and save to cache
    evaluations = download_benchmark_evaluations(
        benchmark_id=99,  # OpenML-CC18
        metrics=['predictive_accuracy', 'area_under_roc_curve'],
        cache_file='results/openml_cc18_evaluations.pkl'
    )
    
    # Print summary
    print("\nSummary of downloaded evaluations:")
    print(f"Downloaded data for {len(evaluations)} tasks")
    
    # Convert to DataFrame for easier analysis
    summary_rows = []
    for task_id, task_data in evaluations.items():
        for metric in ['predictive_accuracy', 'area_under_roc_curve']:
            if metric in task_data:
                row = {
                    'task_id': task_id,
                    'dataset_name': task_data.get('dataset_name', f"Unknown-{task_id}"),
                    'metric': metric,
                    'mean': task_data[metric]['mean'],
                    'std': task_data[metric]['std'],
                    'max': task_data[metric]['max'],
                    'min': task_data[metric]['min'],
                    'count': task_data[metric]['count']
                }
                summary_rows.append(row)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('results/openml_cc18_evaluations_summary.csv', index=False)
        print(f"Saved summary to results/openml_cc18_evaluations_summary.csv")
        
        # Print some basic stats
        print("\nBasic statistics by metric:")
        print(summary_df.groupby('metric')[['mean', 'std', 'count']].describe())
