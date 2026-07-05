import numpy as np
import json
from tabpfn import TabPFNRegressor
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tabpfn_final_accuracy.utils.arch_features as arch_features

parser = argparse.ArgumentParser(description='Parser parsing arguments for the evolutionary NAS algorithm')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
parser.add_argument('--datafile', type=str, default='tabpfn_final_accuracy/data/merged_data.json', help='Path to the JSON file containing architecture data')
parser.add_argument('--arch_str_to_index_dict_file', type=str, default='tabpfn_final_accuracy/data/arch_str_to_index.json', help='Path to the JSON file containing the architecture string to index mapping')

# the individual:
# arch number, for how long it was trained, final (predicated) accuracy

def initialize_population(size, data):
    archs = np.random.choice(15625, size=size, replace=False)
    return [(arch, 200, np.float64(data[arch]["testacc"])) for arch in archs]

def select_parents(population, num_parents):
    # select parents according to roulette wheel selection
    total_fitness = sum(individual[2] for individual in population)
    assert total_fitness > 0, "Total fitness is zero, cannot perform selection"
    probabilities = [individual[2] / total_fitness for individual in population]
    selected_indices = np.random.choice(len(population), size=num_parents, replace=False, p=probabilities)
    selected_parents = [population[i] for i in selected_indices]
    return selected_parents

def mutate_architecture(arch, mut_prob, data, arch_str_to_index):
    # mutate the architecture by randomly changing edges with mut_prob probability
    arch_str = data[arch]["arch"]
    arch_list = arch_features.arch_to_list(arch_str)
    previous_arch_list = arch_list.copy()

    for edge in range(len(arch_list)):
        if np.random.rand() < mut_prob:
            new_value = np.random.randint(0, 4)
            arch_list[edge] = new_value

    if arch_list == previous_arch_list:
        # if no mutation happened, we randomly mutate one edge
        edge = np.random.randint(0, len(arch_list))
        new_value = np.random.randint(0, 4)
        arch_list[edge] = new_value

    # convert back to index
    new_arch_inx = arch_features.get_arch_index_from_edges(arch_list, arch_str_to_index)
    return new_arch_inx

def tabpfn_predict(archs_to_predict, population, data, first_n_epochs_features):
    # get features and labels for training architectures
    train_features = []
    train_labels = []
    for ind in population:
        if ind[1] < first_n_epochs_features:
            continue
        features, label = arch_features.get_features_and_label(data, ind[0], edge_features="raw", zero_cost_features=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'], first_n_epochs_features=first_n_epochs_features)
        train_features.append(features)
        train_labels.append(label)

    # get features for the architectures to predict
    pred_features = []
    true_labels_for_printing_only = []
    for arch in archs_to_predict:
        feature, true_label_not_used_for_training = arch_features.get_features_and_label(data, arch, edge_features="raw", zero_cost_features=['f', 'g_n', 'g', 'j_c', 'p', 's', 'sy', 'sy_bn'], first_n_epochs_features=first_n_epochs_features)
        pred_features.append(feature)
        true_labels_for_printing_only.append(true_label_not_used_for_training)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    pred_features = np.array(pred_features)

    # if we only have one architecture to predict, we need to reshape the features
    if len(pred_features.shape) == 1:
        pred_features = pred_features.reshape(1, -1)

    # fit tabpfn model
    model = TabPFNRegressor()
    model.fit(train_features, train_labels)

    # predict accuracy for the given architecture
    pred_acc = model.predict(pred_features)

    print(f"Predictions and true labels after training for {first_n_epochs_features} epochs:")
    for i in range(len(archs_to_predict)):
        print(f"arch {archs_to_predict[i]}: last test accuracy: {data[archs_to_predict[i]]['accuracies'][str(first_n_epochs_features)]['test_accuracy']}, prediction: {pred_acc[i]}, true label: {true_labels_for_printing_only[i]}")
    return pred_acc

def generate_offspring(population, num_parents, num_offspring, mut_prob, data, arch_str_to_index):
    """
    Generates offspring architectures by mutating the parent architectures.

    Args:
        population (list): The current population of architectures, where each architecture is represented as a tuple (arch_index, trained_epochs, predicted_accuracy).
        num_parents (int): The number of parent architectures to select for mutation.
        num_offspring (int): The number of offspring architectures to generate.
        mut_prob (float): The probability of mutation for each edge in the architecture.
        data (dict): The dataset containing information about each architecture.
        arch_str_to_index (dict): A dictionary mapping architecture strings to their corresponding indices.

    Returns:
        np.array: An array of offspring architecture indices generated through mutation.
    """

    population_archs = [ind[0] for ind in population]
    # Select best-performing architectures as parents
    parents = select_parents(population, num_parents)

    # Generate offspring through mutation
    offspring_archs = []
    which_parent_index = 0
    while len(offspring_archs) < num_offspring:
        parent = parents[which_parent_index % num_parents]
        child_arch = mutate_architecture(parent[0], mut_prob, data, arch_str_to_index)
        
        if child_arch not in population_archs and child_arch not in offspring_archs:
            # we only add the child architecture if it is not already in the population or in the offspring
            offspring_archs.append(child_arch)
        which_parent_index += 1
        
    offspring_archs = np.array(offspring_archs)

    return offspring_archs

def split_offspring_to_promising_and_finalized(offspring_archs, offsprings_predicted_acc_dict, children_predicted_acc, best_individual, architectures_trained_for_epochs, current_treshold, prediction_variance_treshold):
    """
    Splits the offspring architectures into promising and finalized.

    The criterion for being finalized is:
        - If the predicted accuracy of the architecture is lower than the best individual in the population minus a certain threshold
        - If the predicted accuracy is stable for a certain number of epochs, meaning it does not change much in the last epochs
    Such architectures are added to the finalized children list, which will not be trained further.
    Others are kept in the offspring list, which will be trained further.

    args:
        offspring_archs: list of offspring architectures to split
        offsprings_predicted_acc_dict: dictionary mapping offspring architectures to a list of their predicted accuracies in each epoch of training
        children_predicted_acc: list of the latest predicted accuracies for the offspring architectures
        best_individual: the best individual in the population
        current_treshold: the current threshold for being promising, which can change during training
        prediction_variance_treshold: the threshold for considering the prediction stable, meaning it does not change much in the last epochs
        architectures_trained_for_epochs: how many epochs the offspring architectures have been trained for

    returns:
        finalized_children: list of architectures that are finalized and will not be trained further
        offspring_archs: list of architectures that are still promising and will be trained further
    """
    
    finalized_children = []
    for i in range(len(offspring_archs)):
        offsprings_predicted_acc_dict[offspring_archs[i]].append(children_predicted_acc[i])
        no_longer_promising_condition = children_predicted_acc[i] <= (best_individual[2] - current_treshold)
        prediction_stable_condition = False
        if architectures_trained_for_epochs > 10:
            # if the prediction has small changes in the last 10 epochs, we consider it stable
            prediction_stable_condition = np.std(offsprings_predicted_acc_dict[offspring_archs[i]][-10:]) < prediction_variance_treshold
        if no_longer_promising_condition or prediction_stable_condition:
            # if the architecture is no longer promising, or the prediction is stable, we finalize it
            finalized_children.append((offspring_archs[i], architectures_trained_for_epochs, children_predicted_acc[i] if children_predicted_acc[i] > 0 else 0))
            offspring_archs[i] = -1  # mark for deletion

    # remove architectures marked for deletion
    offspring_archs = offspring_archs[offspring_archs != -1]

    return finalized_children, offspring_archs


def evolutionary_algorithm(num_offspring, num_generations, mut_prob, init_population_size, num_parents, data, arch_str_to_index, treshold_to_train_further, prediction_variance_treshold = 0.1):
    # Initialize the population with random architectures
    population = initialize_population(init_population_size, data)

    for generation in range(num_generations):
        print(f"Generation {generation}:")

        best_individual = max(population, key=lambda x: x[2])
        print(f"Best individual in the population: arch {best_individual[0]}, trained for {best_individual[1]} epochs, accuracy: {best_individual[2]}")

        offspring_archs = generate_offspring(population, num_parents, num_offspring, mut_prob, data, arch_str_to_index)

        architectures_trained_for_epochs = 0
        children_predicted_acc = tabpfn_predict(offspring_archs, population, data, architectures_trained_for_epochs)

        # we keep track of the predicted accuracies for each offspring architecture in each epoch of training
        # this is needed to decide if the architecture is promising or not, and if we should train it further or not
        offsprings_predicted_acc_dict = dict()
        for i in range(len(offspring_archs)):
            offsprings_predicted_acc_dict[offspring_archs[i]] = [children_predicted_acc[i]]

        # we split the offspring architectures into promising and finalized
        # finalized architectures are added to the finalized children list, which will not be trained further
        finalized_children, offspring_archs = split_offspring_to_promising_and_finalized(offspring_archs, offsprings_predicted_acc_dict, children_predicted_acc, best_individual, architectures_trained_for_epochs, treshold_to_train_further, prediction_variance_treshold)

        # we will lower the treshold for being promising during training
        treshold = treshold_to_train_further

        # we keep training the promising architectures until they are finalized
        while len(offspring_archs) > 0 and architectures_trained_for_epochs < 199:
            # only if the architecture looks promising, we continue training
            architectures_trained_for_epochs += 1
            children_predicted_acc = tabpfn_predict(offspring_archs, population, data, architectures_trained_for_epochs)

            # we check for each architecture if it is still promising, or if we don't need to train it further
            new_finalized_children, offspring_archs = split_offspring_to_promising_and_finalized(offspring_archs, offsprings_predicted_acc_dict, children_predicted_acc, best_individual, architectures_trained_for_epochs, treshold, prediction_variance_treshold)
            finalized_children.extend(new_finalized_children)

            # we lower the treshold for further training,
            # so that we are more strict with architectures that are already trained for a long time
            treshold -= treshold_to_train_further/200

        # if there are still architectures that are perspective after training for 199 epochs
        # we add them to the finalized children, since we don't train for more than 200 epochs        
        if architectures_trained_for_epochs == 199:
            for i in range(len(offspring_archs)):
                # training is finished, we get the final accuracy from the data
                final_acc = data[offspring_archs[i]]["testacc"]
                finalized_children.append((offspring_archs[i], 200, final_acc))

        population.extend(finalized_children)

    best_individual = max(population, key=lambda x: x[2])
    print(f"Best individual after {num_generations} generations: arch {best_individual[0]}, trained for {best_individual[1]} epochs, predicted accuracy: {best_individual[2]}")
    return best_individual, population


def main(args: argparse.Namespace):
    # set random seed
    np.random.seed(args.seed)

    # check if datafile exists
    if not os.path.exists(args.datafile):
        print("Error: Data file does not exist.")
        return
    
    with open(args.datafile, 'r') as f:
        data = json.load(f)

    with open(args.arch_str_to_index_dict_file, 'r') as f:
        arch_str_to_index = json.load(f)
    
    # mut prob should be 1/6 to happen approximately one mutation per architecture, since there are 6 edges to mutate
    # num parents should be around 10-20% of the init population size, to have enough diversity but also good performance

    best_individual, population = evolutionary_algorithm(num_offspring=20, num_generations=10, mut_prob=1/6, init_population_size=20, num_parents=4, data=data, arch_str_to_index=arch_str_to_index, treshold_to_train_further=3, prediction_variance_treshold=0.1)

    # now we will "train" the best individual for 200 epochs and see how good it is
    best_arch = best_individual[0]
    print(f"Best architecture {best_arch} accuracy after training for 200 epochs: {data[best_arch]['testacc']}")

    # now we will check wether the best architecture is truly the best one from the population
    best_ind_arch_from_population = best_arch
    achieved_accuracy = data[best_arch]['testacc']
    for ind in population:
        ind_arch = ind[0]
        ind_acc = data[ind_arch]['testacc']
        if ind_acc > achieved_accuracy:
            best_ind_arch_from_population = ind_arch
            achieved_accuracy = ind_acc

    if best_ind_arch_from_population != best_arch:
        print(f"Best architecture from population is {best_ind_arch_from_population} with accuracy {achieved_accuracy}, which is better than the best individual from the evolutionary algorithm.")
    else:
        print(f"Best architecture from population is the same as the best individual from the evolutionary algorithm, with accuracy {achieved_accuracy}.")
        print("Algorithm was successful in finding the best architecture in the searched part of the population.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
