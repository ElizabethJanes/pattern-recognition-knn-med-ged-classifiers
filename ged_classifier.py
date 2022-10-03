import numpy as np
from classifier_utilities import get_class_zero_training_data, get_class_one_training_data


def ged_classifier(training_dataset, training_labels, test_data_point, class_zero_prototype, class_one_prototype):
    class_zero_training_data = get_class_zero_training_data(training_dataset, training_labels)
    class_one_training_data = get_class_one_training_data(training_dataset, training_labels)

    class_zero_ged = compute_general_euclidean_distance(class_zero_training_data, class_zero_prototype, test_data_point)
    class_one_ged = compute_general_euclidean_distance(class_one_training_data, class_one_prototype, test_data_point)

    if class_zero_ged < class_one_ged:
        return 0
    else:
        return 1


def compute_general_euclidean_distance(class_training_data, class_prototype, test_data_point_vector):
    sigma_inverse = get_sigma_inverse(class_training_data)
    diff = test_data_point_vector - class_prototype
    ged = (diff.T @ sigma_inverse @ diff)**(1/2)
    return ged


def ged_decision_boundary_coefficients(class_zero_prototype, class_one_prototype, training_data, training_labels):
    class_zero_training_data = get_class_zero_training_data(training_data, training_labels)
    class_zero_sigma_inverse = get_sigma_inverse(class_zero_training_data)

    class_one_training_data = get_class_one_training_data(training_data, training_labels)
    class_one_sigma_inverse = get_sigma_inverse(class_one_training_data)

    q0 = class_zero_sigma_inverse - class_one_sigma_inverse
    q1 = 2*(np.dot(class_one_prototype, class_one_sigma_inverse) - np.dot(class_zero_prototype, class_zero_sigma_inverse))
    print(np.dot(class_zero_prototype, class_zero_sigma_inverse))
    print(class_zero_prototype)
    q2 = (np.dot(np.dot(class_zero_prototype, class_zero_sigma_inverse), class_zero_prototype)) - (np.dot(np.dot(class_one_prototype, class_one_sigma_inverse), class_one_prototype))

    return q0, q1, q2


def get_sigma_inverse(class_training_data):
    sigma = np.cov(class_training_data.transpose())  # covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    eigenvalues_inverse = 1 / eigenvalues
    delta_inverse = np.diag(eigenvalues_inverse)
    sigma_inverse = eigenvectors @ delta_inverse @ eigenvectors.transpose()

    return sigma_inverse


def ged_classifier_2d(test_data_point, class_zero_prototype, class_one_prototype, training_data, training_labels):
    class_zero_training_data = get_class_zero_training_data(training_data, training_labels)
    class_one_training_data = get_class_one_training_data(training_data, training_labels)

    class_zero_eigenvectors = get_class_zero_eigenvectors()
    class_one_eigenvectors = get_class_one_eigenvectors()

    class_zero_ged = compute_2d_general_euclidean_distance(
        class_zero_training_data, class_zero_prototype, test_data_point, class_zero_eigenvectors
    )
    class_one_ged = compute_2d_general_euclidean_distance(
        class_one_training_data, class_one_prototype, test_data_point, class_one_eigenvectors
    )

    if class_zero_ged < class_one_ged:
        return 0
    else:
        return 1


def compute_2d_general_euclidean_distance(class_training_data, class_prototype, test_data_point_vector, eigenvectors):
    sigma = np.cov(class_training_data.transpose())

    # temporary!
    # real_eigenvalues, real_eigenvectors = np.linalg.eig(sigma)
    # print(real_eigenvectors)

    # compute eigenvalues of sigma
    determinant_expansion = np.polynomial.polynomial.Polynomial(
        coef=[(sigma[0, 0] * sigma[1, 1]) - (sigma[0, 1] * sigma[1, 0]), (-sigma[1, 1] - sigma[0, 0]), 1]
    )
    eigenvalues = determinant_expansion.roots()

    eigenvalues_inverse = 1 / eigenvalues
    delta_inverse = np.diag(eigenvalues_inverse)
    sigma_inverse = eigenvectors @ delta_inverse @ eigenvectors.transpose()

    diff = test_data_point_vector - class_prototype
    ged = (diff.T @ sigma_inverse @ diff)**(1/2)

    return ged


def get_class_zero_eigenvectors():
    # Eigenvectors were computed by hand
    zero_eigenvectors = np.array([[-0.5861, -1.638], [0.8102, 0.9986]])
    return zero_eigenvectors


def get_class_one_eigenvectors():
    # Eigenvectors were computed by hand
    one_eigenvectors = np.array([[0.9997, -0.0239], [0.0239, 0.9997]])
    return one_eigenvectors
