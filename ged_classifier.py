import numpy as np
import matplotlib.pyplot as plt
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


def ged_decision_boundary_function(q0, q1, q2, x1, x2):
    x = np.array([x1, x2])
    boundary_point = np.dot(np.dot(x, q0), x) + np.dot(q1, x) + q2
    return boundary_point


def plot_2d_ged_decision_boundary(training_data, training_labels, q0, q1, q2):
    class_zero_indices = np.where(training_labels == 0)
    class_one_indices = np.where(training_labels == 1)
    class_zero_data = training_data[class_zero_indices]
    class_one_data = training_data[class_one_indices]

    step = 1
    x1 = np.arange(-1500, 2500, step)
    x2 = np.arange(-1000, 2000, step)

    function_values = np.ndarray((x2.shape[0], x1.shape[0]), dtype=np.float32)

    for i in range(x2.shape[0]):
        for j in range(x1.shape[0]):
            function_val = ged_decision_boundary_function(
                q0, q1, q2, x1[j], x2[i]
            )
            function_values[i, j] = function_val

    ha, = plt.plot(class_zero_data[:, 0], class_zero_data[:, 1], 'r.', label='Class Zero')
    hb, = plt.plot(class_one_data[:, 0], class_one_data[:, 1], 'b.', label='Class One')
    ctr = plt.contour(x1, x2, function_values, levels=(0,), colors='k')
    h_bnd, _ = ctr.legend_elements()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend([ha, hb, h_bnd[0]], ['Class Zero', 'Class One', 'Classifier Boundary'])
    plt.title(f'2D GED Classifier Decision Boundary')

    plt.show()
