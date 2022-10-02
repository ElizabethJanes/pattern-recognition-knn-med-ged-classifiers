import numpy as np
import matplotlib.pyplot as plt
from knn_classifier import knn_classifier
from med_classifier import med_classifier
from ged_classifier import ged_classifier


def plot_knn_prediction_error(pca_test_data, test_labels, pca_training_data, training_labels):
    incorrect_prediction_count = np.zeros(5)
    error = np.empty(5)
    for k in range(1, 6):
        print(k)
        for test_point, test_label in zip(pca_test_data, test_labels):
            test_classification = knn_classifier(k, pca_training_data, training_labels, test_point)
            if test_classification != test_label:
                incorrect_prediction_count[k-1] += 1
                print('incorrect prediction')
                print(incorrect_prediction_count[k-1])
        error[k-1] = incorrect_prediction_count[k-1]/len(pca_test_data)
        print(f'error = {error[k-1]} when k = {k}')

    # Question 2: Plot the test set error for each value of k
    k = np.array([1, 2, 3, 4, 5])
    plt.scatter(k, error)
    plt.xlabel('k')
    plt.ylabel('Prediction Error')
    plt.title('kNN Classifier Prediction Error vs. k')
    plt.show()


def plot_med_prediction_error(pca_test_data, test_labels, class_zero_prototype, class_one_prototype):
    incorrect_prediction_count = 0
    for test_point, test_label in zip(pca_test_data, test_labels):
        test_classification = med_classifier(test_point, class_zero_prototype, class_one_prototype)
        if test_classification != test_label:
            incorrect_prediction_count += 1
    error = incorrect_prediction_count/len(pca_test_data)
    print(f'med classification error = {error}')

    # Question 2: Plot the test set error
    k = np.array(['MED'])
    plt.scatter(k, error)
    plt.xlabel('MED')
    plt.ylabel('Prediction Error')
    plt.title('MED Classifier Prediction Error')
    plt.show()


def plot_ged_prediction_error(pca_test_data, test_labels, pca_training_data, training_labels, class_zero_prototype, class_one_prototype):
    incorrect_prediction_count = 0
    for test_point, test_label in zip(pca_test_data, test_labels):
        test_classification = ged_classifier(pca_training_data, training_labels, test_point, class_zero_prototype, class_one_prototype)
        if test_classification != test_label:
            incorrect_prediction_count += 1
    error = incorrect_prediction_count/len(pca_test_data)
    print(f'ged classification error = {error}')

    # Question 2: Plot the test set error
    k = np.array(['GED'])
    plt.scatter(k, error)
    plt.xlabel('GED')
    plt.xlabel('GED')
    plt.ylabel('Prediction Error')
    plt.title('GED Classifier Prediction Error')
    plt.show()
