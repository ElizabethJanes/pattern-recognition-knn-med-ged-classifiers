import numpy as np
import matplotlib.pyplot as plt
from EuclideanDistance import EuclideanDistance


def knn_classifier(k, pca_training_data, training_labels, test_data_point):

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

    # calculate the number of occurrences of each class out of k nearest neighbours
    for neighbour in k_nearest_neighbours:
        if neighbour.training_vector_class_label == 0:
            class_zero += 1
        elif neighbour.training_vector_class_label == 1:
            class_one += 1

    # classify test data point based on most frequent class
    if class_zero > class_one:
        return 0
    else:
        return 1


def plot_knn_decision_boundaries(training_data, training_labels):

    class_zero_indices = np.where(training_labels == 0)
    class_one_indices = np.where(training_labels == 1)
    class_zero_data = training_data[class_zero_indices]
    class_one_data = training_data[class_one_indices]

    step = 50
    x1 = np.arange(-1500, 2500, step)
    x2 = np.arange(-1000, 2000, step)

    for k in range(1, 6):

        classification_array = np.ndarray((x2.shape[0], x1.shape[0]), dtype=np.uint8)

        for i in range(x2.shape[0]):
            for j in range(x1.shape[0]):
                test_point = np.array([x1[j], x2[i]])
                test_class = knn_classifier(k, training_data, training_labels, test_point)
                classification_array[i, j] = test_class

        ha, = plt.plot(class_zero_data[:, 0], class_zero_data[:, 1], 'r.', label='Class Zero')
        hb, = plt.plot(class_one_data[:, 0], class_one_data[:, 1], 'b.', label='Class Zero')

        contour = plt.contour(x1, x2, classification_array, colors='k')

        h_bnd, _ = contour.legend_elements()
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.legend([ha, hb, h_bnd[0]], ['Class Zero', 'Class One', 'Classifier Boundary'])
        plt.title(f'kNN Classifier Decision Boundary for k = {k}')

        plt.show()
