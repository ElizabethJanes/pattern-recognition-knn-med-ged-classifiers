import matplotlib.pyplot as plt
import numpy as np
from knn_classifier import knn_classifier


def plot_training_data(training_data, training_labels):
    class_zero_indices = np.where(training_labels == 0)
    class_one_indices = np.where(training_labels == 1)
    class_zero_data = training_data[class_zero_indices]
    class_one_data = training_data[class_one_indices]

    plt.scatter(class_zero_data[:, 0], class_zero_data[:, 1])
    plt.scatter(class_one_data[:, 0], class_one_data[:, 1])

    plt.show()


def plot_knn_decision_boundaries(training_data, training_labels):

    class_zero_indices = np.where(training_labels == 0)
    class_one_indices = np.where(training_labels == 1)
    class_zero_data = training_data[class_zero_indices]
    class_one_data = training_data[class_one_indices]

    print('decision boundaries')
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
