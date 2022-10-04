import numpy as np
import matplotlib.pyplot as plt
from knn_classifier import knn_classifier
from med_classifier import med_classifier
from ged_classifier import ged_classifier, ged_classifier_2d


def plot_knn_prediction_error(pca_test_data, test_labels, pca_training_data, training_labels):
    incorrect_prediction_count = np.zeros(5)
    true_class_zero = np.zeros(5)
    true_class_one = np.zeros(5)
    false_class_zero = np.zeros(5)
    false_class_one = np.zeros(5)
    error = np.empty(5)
    for k in range(1, 6):
        print(k)
        for test_point, test_label in zip(pca_test_data, test_labels):
            test_classification = knn_classifier(k, pca_training_data, training_labels, test_point)
            if test_classification != test_label:
                incorrect_prediction_count[k-1] += 1
                if test_classification == 0:
                    false_class_zero[k-1] += 1
                else:
                    false_class_one[k-1] += 1
            else:
                if test_classification == 0:
                    true_class_zero[k-1] += 1
                else:
                    true_class_one[k-1] += 1
        error[k-1] = incorrect_prediction_count[k-1]/len(pca_test_data)
        print(f'error = {error[k-1]} when k = {k}')
        print(f'kNN Classifier Confusion Matrix Results, k = {k}')
        print(f'True Class Zero: {true_class_zero[k-1]}')
        print(f'True Class One: {true_class_one[k-1]}')
        print(f'False Class Zero: {false_class_zero[k-1]}')
        print(f'False Class One: {false_class_one[k-1]}')

    # Question 2: Plot the test set error for each value of k
    k = np.array([1, 2, 3, 4, 5])
    plt.bar(k, error)
    plt.xlabel('k')
    plt.ylabel('Prediction Error')
    plt.title('kNN Classifier Prediction Error vs. k')
    plt.show()


def med_prediction_error(pca_test_data, test_labels, class_zero_prototype, class_one_prototype):
    incorrect_prediction_count = 0
    true_class_zero = 0
    true_class_one = 0
    false_class_zero = 0
    false_class_one = 0
    for test_point, test_label in zip(pca_test_data, test_labels):
        test_classification = med_classifier(test_point, class_zero_prototype, class_one_prototype)
        if test_classification != test_label:
            incorrect_prediction_count += 1
            if test_classification == 0:
                false_class_zero += 1
            else:
                false_class_one += 1
        else:
            if test_classification == 0:
                true_class_zero += 1
            else:
                true_class_one += 1
    error = incorrect_prediction_count/len(pca_test_data)
    print(f'MED classification error = {error}')
    print(f'MED Classifier Confusion Matrix Results')
    print(f'True Class Zero: {true_class_zero}')
    print(f'True Class One: {true_class_one}')
    print(f'False Class Zero: {false_class_zero}')
    print(f'False Class One: {false_class_one}')

    return true_class_zero, true_class_one, false_class_zero, false_class_one, error


def ged_prediction_error(
        pca_test_data,
        test_labels,
        pca_training_data,
        training_labels,
        class_zero_prototype,
        class_one_prototype
):
    incorrect_prediction_count = 0
    true_class_zero = 0
    true_class_one = 0
    false_class_zero = 0
    false_class_one = 0
    for test_point, test_label in zip(pca_test_data, test_labels):
        test_classification = ged_classifier(
            pca_training_data, training_labels, test_point, class_zero_prototype, class_one_prototype
        )
        if test_classification != test_label:
            incorrect_prediction_count += 1
            if test_classification == 0:
                false_class_zero += 1
            else:
                false_class_one += 1
        else:
            if test_classification == 0:
                true_class_zero += 1
            else:
                true_class_one += 1
    error = incorrect_prediction_count/len(pca_test_data)
    print(f'GED classification error = {error}')
    print(f'GED Classifier Confusion Matrix Results')
    print(f'True Class Zero: {true_class_zero}')
    print(f'True Class One: {true_class_one}')
    print(f'False Class Zero: {false_class_zero}')
    print(f'False Class One: {false_class_one}')

    return true_class_zero, true_class_one, false_class_zero, false_class_one, error


def ged_2d_prediction_error(
        pca_test_data,
        test_labels,
        pca_training_data,
        training_labels,
        class_zero_prototype,
        class_one_prototype
):
    incorrect_prediction_count = 0
    true_class_zero = 0
    true_class_one = 0
    false_class_zero = 0
    false_class_one = 0
    for test_point, test_label in zip(pca_test_data, test_labels):
        test_classification = ged_classifier_2d(
            test_point, class_zero_prototype, class_one_prototype, pca_training_data, training_labels
        )
        if test_classification != test_label:
            incorrect_prediction_count += 1
            if test_classification == 0:
                false_class_zero += 1
            else:
                false_class_one += 1
        else:
            if test_classification == 0:
                true_class_zero += 1
            else:
                true_class_one += 1
    error = incorrect_prediction_count / len(pca_test_data)
    print(f'2D GED classification error = {error}')
    print(f'GED Classifier Confusion Matrix Results')
    print(f'True Class Zero: {true_class_zero}')
    print(f'True Class One: {true_class_one}')
    print(f'False Class Zero: {false_class_zero}')
    print(f'False Class One: {false_class_one}')

    return true_class_zero, true_class_one, false_class_zero, false_class_one, error


def plot_med_ged_prediction_error(
        med_incorrect_class_zero_count,
        med_incorrect_class_one_count,
        total_med_prediction_error,
        ged_incorrect_class_zero_count,
        ged_incorrect_class_one_count,
        total_ged_prediction_error
):
    x = ['MED Class Zero Error',
         'MED Class One Error',
         'MED Total Error',
         'GED Class Zero Error',
         'GED Class One Error',
         'GED Total Error']
    y = [med_incorrect_class_zero_count/980,
         med_incorrect_class_one_count/1135,
         total_med_prediction_error,
         ged_incorrect_class_zero_count/980,
         ged_incorrect_class_one_count/1135,
         total_ged_prediction_error]

    plt.bar(x, y)
    plt.xlabel('Class')
    plt.ylabel('Prediction Error')
    plt.title('Minimum Euclidean Distance and General Euclidean Distance Classifier Prediction Error')
    plt.show()
