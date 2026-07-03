import argparse
import numpy as np
import os
import json

from tabpfn_with_nats_bench.utils.dataloader import download_data_if_not_exists, load_predictions_file, get_true_labels
from tabpfn_with_nats_bench.utils.results_utils import plot_two_predictions_vs_true

parser = argparse.ArgumentParser(description="Analyze the results of the experiments. And create plots for the results.")
parser.add_argument("--datafile", type=str, default="tabpfn_with_nats_bench/data/merged_data.json", help="Path to the JSON file containing architecture data.")
parser.add_argument("--logdir", type=str, default="tabpfn_with_nats_bench/logs/analysis_results", help="Directory to save plots")
parser.add_argument("--pred_file_1", type=str, default="tabpfn_with_nats_bench/logs/grid_search_results/seed_42/train-1000_cf-True_aif-False_ef-raw_noZC_noSkipUnconnected_enf-0/predictions.txt", help="Path to the first predictions file.")
parser.add_argument("--pred_file_2", type=str, default="tabpfn_with_nats_bench/logs/grid_search_results/seed_42/train-1000_cf-False_aif-False_ef-raw_noZC_skipUnconnected_enf-0/predictions.txt", help="Path to the second predictions file.")
parser.add_argument("--title1", type=str, default="Predikce s příznakem spojitosti", help="Title for the first predictions plot.")
parser.add_argument("--title2", type=str, default="Predikce s odstraněnými nespojitými architekturami", help="Title for the second predictions plot.")
parser.add_argument("--plot_name", type=str, default="Connected_Feature_vs_skipUnconnected.pdf", help="Name of the plot file to save.")


if __name__ == "__main__":
    args = parser.parse_args()

    # load the two files
    indices_1, predictions_1 = load_predictions_file(args.pred_file_1)
    indices_2, predictions_2 = load_predictions_file(args.pred_file_2)
    
    # load the datafile
    download_data_if_not_exists(args.datafile)
    with open(args.datafile, "r") as f:
        data = json.load(f)

    # get the true labels for the first predictions file
    true_labels_1 = get_true_labels(indices_1, data)

    # check if the indices are the same
    if not np.array_equal(indices_1, indices_2):
        print("Warning: The indices in the two prediction files are not the same.")
        true_labels_2 = get_true_labels(indices_2, data)
    else:
        true_labels_2 = true_labels_1

    # create the logdir if it does not exist
    if args.logdir is not None:
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        path_to_save = os.path.join(args.logdir, args.plot_name)
    else:
        path_to_save = None
    
    # plot the two predictions vs the true labels
    plot_two_predictions_vs_true(true_labels_1, true_labels_2, predictions_1, predictions_2, title1=args.title1, title2=args.title2, path_to_save=path_to_save)

    



    

    



