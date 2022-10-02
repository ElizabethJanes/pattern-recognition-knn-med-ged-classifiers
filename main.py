from dataset_utilities import get_2d_dataset, get_20d_dataset
from knn_classifier import knn_classifier
from plot import plot_knn_decision_boundaries
from prediction_error import plot_knn_prediction_error, plot_med_prediction_error, plot_ged_prediction_error
from med_classifier import med_classifier, med_decision_boundary_coefficients
from classifier_utilities import get_class_prototypes
from ged_classifier import ged_classifier, ged_decision_boundary_coefficients

if __name__ == '__main__':
    pca_2d_training_data, training_labels_2d, pca_2d_test_data, test_labels_2d = get_2d_dataset()
    pca_20d_training_data, training_labels_20d, pca_20d_test_data, test_labels_20d = get_20d_dataset()
    class_zero_prototype, class_one_prototype = get_class_prototypes(pca_20d_training_data, training_labels_20d)

    # Exercise 1: Nearest Neighbour Classifier
    # Question 1: Implement kNN Classifier, compute kNN solution for k = [1, 5]
    print('k-Nearest Neighbour Classifier')
    for k in range(1, 6):
        knn_test_classification = knn_classifier(
            k, pca_2d_training_data, training_labels_2d, pca_2d_test_data[1]
        )
        print(f'When k = {k}, the test point is classified as {knn_test_classification} and the expected '
              f'classification is {test_labels_2d[1]}')

    # Question 1: Plot the classification boundaries between the two classes for k = [1, 5]
    plot_knn_decision_boundaries(pca_2d_training_data, training_labels_2d)

    # Question 2: Make label predictions for all test vectors and find the prediction error for the kNN Classifier
    plot_knn_prediction_error(pca_2d_test_data, test_labels_2d, pca_2d_training_data, training_labels_2d)

    # Exercise 2: MED and GED Classifiers

    # Question 1: Implement MED Classifier
    print('Minimum Euclidean Distance Classifier')
    med_test_classification = med_classifier(pca_20d_test_data[0], class_zero_prototype, class_one_prototype)
    print(f'The test point is classified as {med_test_classification} and the expected classification '
          f'is {test_labels_20d[0]}')

    # Question 1: Implement GED Classifier
    print('General Euclidean Distance Classifier')
    ged_test_classification = ged_classifier(
        pca_20d_training_data,
        training_labels_20d,
        pca_20d_test_data[0],
        class_zero_prototype,
        class_one_prototype
    )
    print(f'The test point is classified as {ged_test_classification} and the expected classification '
          f'is {test_labels_20d[0]}')

    # Question 1: Determine decision boundaries for MED and GED

    w, wo = med_decision_boundary_coefficients(class_zero_prototype, class_one_prototype)
    print(f'The coefficients of the MED decision boundary are w = {w.T} and wo = {wo}')

    q0, q1, q2 = ged_decision_boundary_coefficients(
        class_zero_prototype, class_one_prototype, pca_20d_training_data, training_labels_20d
    )
    print(f'The coefficients of the GED decision boundary are q0 = {q0}, q1 = {q1}, and q2 = {q2}')

    # Question 2: Make label predictions for all test vectors and plot the prediction error for both classifiers
    plot_med_prediction_error(
        pca_20d_test_data, test_labels_20d, class_zero_prototype, class_one_prototype
    )
    plot_ged_prediction_error(
        pca_20d_test_data,
        test_labels_20d,
        pca_20d_training_data,
        training_labels_20d,
        class_zero_prototype,
        class_one_prototype
    )

    # Question 4: Convert training images to 2x1 vectors and plot decision boundaries for MED and GED

    # Question 5: Find the confusion matrices for MED, GED, and kNN (k = [1, 5]) Classifiers using test datasets

