import numpy as np
import matplotlib.pyplot as plt
from EuclideanDistance import EuclideanDistance


def med_classifier(test_data_point, class_zero_prototype, class_one_prototype):
    class_zero_euclidean_dist = EuclideanDistance(class_zero_prototype, 0, test_data_point)
    class_one_euclidean_dist = EuclideanDistance(class_one_prototype, 1, test_data_point)

    if class_zero_euclidean_dist.euclidean_distance < class_one_euclidean_dist.euclidean_distance:
        return 0
    else:
        return 1


def med_decision_boundary_coefficients(class_zero_prototype, class_one_prototype):
    w = class_zero_prototype - class_one_prototype
    wo = 1/2*(np.dot(class_one_prototype, class_one_prototype) - np.dot(class_zero_prototype, class_zero_prototype))
    return w, wo


def med_decision_boundary_function(w, wo, x1, x2):
    x = np.array([x1, x2])
    boundary_point = np.dot(w, x) + wo
    return boundary_point


def plot_2d_med_decision_boundary(training_data, training_labels, w, wo):
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
            function_val = med_decision_boundary_function(w, wo, x1[j], x2[i])
            function_values[i, j] = function_val

    ha, = plt.plot(class_zero_data[:, 0], class_zero_data[:, 1], 'r.', label='Class Zero')
    hb, = plt.plot(class_one_data[:, 0], class_one_data[:, 1], 'b.', label='Class One')
    ctr = plt.contour(x1, x2, function_values, levels=(0,), colors='k')
    h_bnd, _ = ctr.legend_elements()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend([ha, hb, h_bnd[0]], ['Class Zero', 'Class One', 'Classifier Boundary'])
    plt.title(f'2D MED Classifier Decision Boundary')

    plt.show()
