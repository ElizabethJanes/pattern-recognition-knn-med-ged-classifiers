import numpy as np


class GeneralEuclideanDistance:
    def __init__(self, training_dataset, training_labels, test_vector):
        self.training_data = training_dataset
        self.training_labels = training_labels
        self.test_data_point_vector = test_vector
        self.class_zero_training_data = self.get_class_zero_training_data()
        self.class_one_training_data = self.get_class_one_training_data()
        self.class_zero_ged = self.compute_class_zero_general_euclidean_distance()
        self.class_one_ged = self.compute_class_zero_general_euclidean_distance()

    def compute_class_zero_general_euclidean_distance(self):
        print('compute class zero general euclidean distance')
        class_zero_cov_matrix = np.cov(self.class_zero_training_data.T)  # covariance matrix = sigma
        print(class_zero_cov_matrix.shape)

        eigenvalues, eigenvectors = np.linalg.eig(class_zero_cov_matrix)
        print(eigenvalues.shape)
        print(eigenvectors.shape)
        A = eigenvectors.T
        print(A.shape)
        delta = np.matmul((np.matmul(A, class_zero_cov_matrix)), A.T)
        print(delta.shape)
        print(delta)
        delta_inverse = delta**(-1)

        # ged = self.compute_euclidean_distance()
        ged = 1
        return ged

    def compute_class_one_general_euclidean_distance(self):
        print('compute class one general euclidean distance')
        class_one_cov_matrix = np.cov(self.class_one_training_data.T)
        print(class_one_cov_matrix.shape)

        # ged = self.compute_euclidean_distance()
        ged = 1

        return ged

    def compute_euclidean_distance(self, class_prototype):
        delta = self.test_data_point_vector - class_prototype
        euclidean_distance = (np.dot(delta, delta)) ** (1 / 2)

        return euclidean_distance

    def get_class_zero_training_data(self):
        class_zero_indices = np.where(self.training_labels == 0)
        class_zero_data = self.training_data[class_zero_indices]
        print(class_zero_data.shape)
        return class_zero_data

    def get_class_one_training_data(self):
        class_one_indices = np.where(self.training_labels == 1)
        class_one_data = self.training_data[class_one_indices]
        print(class_one_data.shape)
        return class_one_data
