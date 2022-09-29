import numpy as np


class GeneralEuclideanDistance:
    def __init__(self, training_dataset, training_labels, training_vector, training_vector_class_label, test_vector):
        self.training_data = training_dataset
        self.training_labels = training_labels
        self.training_data_point_vector = training_vector
        self.training_vector_class_label = training_vector_class_label
        self.test_data_point_vector = test_vector
        self.class_zero_training_data = self.get_class_zero_training_data()
        self.class_one_training_data = self.get_class_one_training_data()
        self.euclidean_distance = self.compute_general_euclidean_distance()

    def compute_general_euclidean_distance(self):
        class_zero_cov_matrix = np.cov(self.class_zero_training_data, self.class_zero_training_data)
        print(class_zero_cov_matrix)
        class_one_cov_matrix = np.cov(self.class_one_training_data, self.class_one_training_data)
        print(class_one_cov_matrix)

        delta = self.test_data_point_vector - self.training_data_point_vector
        euclidean_distance = (np.dot(delta, delta))**(1/2)

        return euclidean_distance

    def get_class_zero_training_data(self):
        class_zero_indices = np.where(self.training_labels == 0)
        class_zero_data = self.training_data[class_zero_indices]
        return class_zero_data

    def get_class_one_training_data(self):
        class_one_indices = np.where(self.training_labels == 1)
        class_one_data = self.training_data[class_one_indices]
        return class_one_data
