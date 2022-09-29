import numpy as np


class EuclideanDistance:
    def __init__(self, training_vector, training_vector_class_label, test_vector):
        self.training_data_point_vector = training_vector
        self.training_vector_class_label = training_vector_class_label
        self.test_data_point_vector = test_vector
        self.euclidean_distance = self.compute_euclidean_distance()

    def compute_euclidean_distance(self):
        delta = self.test_data_point_vector - self.training_data_point_vector
        euclidean_distance = (np.dot(delta, delta))**(1/2)

        return euclidean_distance
