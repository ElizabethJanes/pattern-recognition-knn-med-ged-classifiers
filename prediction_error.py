import numpy as np
import matplotlib.pyplot as plt
from knn_classifier import knn_classifier
from med_classifier import med_classifier


def plot_knn_prediction_error(pca_test_data, test_labels, pca_training_data, training_labels):
    # incorrect_prediction_count = np.zeros(5)
    # error = np.empty(5)
    # for k in range(1, 6):
    #     print(k)
    #     for test_point, test_label in zip(pca_test_data, test_labels):
    #         test_classification = knn_classifier(k, pca_training_data, training_labels, test_point)
    #         if test_classification != test_label:
    #             incorrect_prediction_count[k-1] += 1
    #             print('incorrect prediction')
    #             print(incorrect_prediction_count[k-1])
    #     error[k-1] = incorrect_prediction_count[k-1]/len(pca_test_data)
    #     print(f'error = {error[k-1]} when k = {k}')

    # Temp to save running time! To be removed!
    error = np.array([0.0037825059101654845, 0.005200945626477541, 0.0018912529550827422, 0.0028368794326241137, 0.0014184397163120568])

    # Question 2: Plot the test set error for each value of k
    k = np.array([1, 2, 3, 4, 5])
    plt.scatter(k, error)
    plt.xlabel('k')
    plt.ylabel('Prediction Error')
    plt.title('kNN Classifier Prediction Error vs. k')
    plt.show()


def plot_med_prediction_error(pca_test_data, test_labels, pca_training_data, training_labels):
    # incorrect_prediction_count = 0
    # for test_point, test_label in zip(pca_test_data, test_labels):
    #     test_classification = med_classifier(pca_training_data, training_labels, test_point)
    #     if test_classification != test_label:
    #         incorrect_prediction_count += 1
    #         print('incorrect prediction')
    #         print(incorrect_prediction_count)
    # # TODO: find error by class
    # error = incorrect_prediction_count/len(pca_test_data)
    # print(f'med classification error = {error}')

    # Temp!
    error = 0.004728132387706856

    # Question 2: Plot the test set error
    k = np.array(['MED'])
    plt.scatter(k, error)
    plt.xlabel('MED')
    plt.ylabel('Prediction Error')
    plt.title('MED Classifier Prediction Error')
    plt.show()
