__author__ = 'chris'

import numpy as np

def get_confusion_tableOLD(ground_truth, predicted_labels):
    assert isinstance(ground_truth[0], int), "Type is not int"
    assert isinstance(predicted_labels[0], int), "Type is not int"

    # matrix(num_of real_clusters x clusters_found)
    conf_table = np.zeros((len(set(ground_truth)), len(set(predicted_labels))), dtype="int8")
    real_clusters_set = set(ground_truth)

    real_clust_map = {}
    index = 0
    for c in real_clusters_set:
        if not c in real_clust_map:
            real_clust_map[c] = index
            index += 1

    for current_clust in real_clust_map.keys():
        for i in range(len(predicted_labels)):
            if real_clust_map[ground_truth[i]] == current_clust:
                cluster_found = predicted_labels[i]
                conf_table[current_clust, cluster_found] = conf_table[current_clust, cluster_found] + 1
    return conf_table


def get_confusion_table(ground_truth, predicted_labels):
    assert len(ground_truth) == len(predicted_labels), "Invalid input arguments"
    assert len(ground_truth) > 0, "Invalid input arguments"
    assert isinstance(ground_truth[0], int), "Type is not int"
    assert isinstance(predicted_labels[0], int), "Type is not int"

    # matrix(ground_truth x predicted_labels)
    conf_table = np.zeros((len(set(ground_truth)), len(set(predicted_labels))))
    real_clust = list(set(ground_truth))
    # it's needed because ground truth can have discontinuous cluster set
    clust_to_index = { real_clust[i]: i for i in range(len(real_clust)) }

    for real_clust in clust_to_index.values():
        for i in range(len(predicted_labels)):
            if clust_to_index[ground_truth[i]] == real_clust:
                cluster_found = predicted_labels[i]
                conf_table[real_clust, cluster_found] = conf_table[real_clust, cluster_found] + 1
    return conf_table
