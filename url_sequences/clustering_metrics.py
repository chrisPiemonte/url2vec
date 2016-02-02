__author__ = 'chris'

import numpy as np

def get_confusion_table(real_membership_list, clusters_found_labels):
    
    assert isinstance(real_membership_list[0], int), "Type is not int"
    assert isinstance(clusters_found_labels[0], int), "Type is not int"
    
    # matrix(num_of real_clusters x clusters_found)
    conf_table = np.zeros((len(set(real_membership_list)), len(set(clusters_found_labels))), dtype="int8")
    real_clusters_set = set(real_membership_list)
    
    for current_clust in real_clusters_set:
        for i in range(len(clusters_found_labels)):
            if real_membership_list[i] == current_clust:
                cluster_found = clusters_found_labels[i]
                conf_table[current_clust, cluster_found] = conf_table[current_clust, cluster_found] + 1
    return conf_table