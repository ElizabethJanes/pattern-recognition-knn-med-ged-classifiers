import numpy as np


def get_class_prototypes(training_data, training_labels):

    class_zero_indices = np.where(training_labels == 0)
    class_zero_data = training_data[class_zero_indices]
    class_zero_prototype = get_prototype(class_zero_data)

    class_one_indices = np.where(training_labels == 1)
    class_one_data = training_data[class_one_indices]
    class_one_prototype = get_prototype(class_one_data)

    return class_zero_prototype, class_one_prototype


def get_prototype(class_training_data):
    class_prototype = np.mean(class_training_data, axis=0)
    return class_prototype


def get_class_zero_training_data(training_data, training_labels):
    class_zero_indices = np.where(training_labels == 0)
    class_zero_data = training_data[class_zero_indices]
    return class_zero_data


def get_class_one_training_data(training_data, training_labels):
    class_one_indices = np.where(training_labels == 1)
    class_one_data = training_data[class_one_indices]
    return class_one_data


def compute_euclidean_distance(self, class_prototype):
    delta = self.test_data_point_vector - class_prototype
    euclidean_distance = (np.dot(delta, delta)) ** (1 / 2)

    return euclidean_distance
