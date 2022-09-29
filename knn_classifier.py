import numpy as np
from EuclideanDistance import EuclideanDistance


def knn_classifier(k, pca_training_data, training_labels, test_data_point):
    x = np.empty((k, 2))  # k closest training data points to test point
    y = np.empty(k)  # labels corresponding to x

    class_zero = 0  # count of nearest neighbours in class zero
    class_one = 0  # count of nearest neighbours in class one

    # make array of computed euclidean distances
    computed_euclidean_distances = []
    for (pca_training_data_point, training_label) in zip(pca_training_data, training_labels):
        euclidean_dist = EuclideanDistance(pca_training_data_point, training_label, test_data_point)
        computed_euclidean_distances.append(euclidean_dist)

    # sort calculated euclidean distances from smallest to largest
    sorted_euclidean_distances = sorted(computed_euclidean_distances, key=lambda distance: distance.euclidean_distance)

    k_nearest_neighbours = []
    for index in range(k):
        k_nearest_neighbours.append(sorted_euclidean_distances[index])
        # print(sorted_euclidean_distances[index].euclidean_distance)
        # print(sorted_euclidean_distances[index].training_vector_class_label)

    # calculate the number of occurrences of each class out of k nearest neighbours
    for neighbour in k_nearest_neighbours:
        if neighbour.training_vector_class_label == 0:
            class_zero += 1
        elif neighbour.training_vector_class_label == 1:
            class_one += 1

    # classify test data point based on most frequent q
    if class_zero > class_one:
        return 0
    else:
        return 1
