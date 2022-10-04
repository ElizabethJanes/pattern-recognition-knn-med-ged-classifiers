from dataset_utilities import get_2d_dataset, get_20d_dataset
from knn_classifier import knn_classifier, plot_knn_decision_boundaries
from prediction_error import plot_knn_prediction_error, med_prediction_error, ged_prediction_error, ged_2d_prediction_error, plot_med_ged_prediction_error
from med_classifier import med_classifier, med_decision_boundary_coefficients, plot_2d_med_decision_boundary
from classifier_utilities import get_class_prototypes
from ged_classifier import ged_classifier, ged_decision_boundary_coefficients, ged_classifier_2d, plot_2d_ged_decision_boundary

if __name__ == '__main__':
    pca_2d_training_data, training_labels_2d, pca_2d_test_data, test_labels_2d = get_2d_dataset()
    pca_20d_training_data, training_labels_20d, pca_20d_test_data, test_labels_20d = get_20d_dataset()
    class_zero_prototype_2d, class_one_prototype_2d = get_class_prototypes(pca_2d_training_data, training_labels_2d)
    class_zero_prototype, class_one_prototype = get_class_prototypes(pca_20d_training_data, training_labels_20d)

    # Exercise 1: Nearest Neighbour Classifier
    # Question 1: Implement kNN Classifier, compute kNN solution for k = [1, 5]
    print('k-Nearest Neighbour Classifier: implemented in function knn_classifier()')

    # Question 1: Plot the classification boundaries between the two classes for k = [1, 5]
    plot_knn_decision_boundaries(pca_2d_training_data, training_labels_2d)

    # Question 2: Make label predictions for all test vectors and find the prediction error for the kNN Classifier
    plot_knn_prediction_error(pca_2d_test_data, test_labels_2d, pca_2d_training_data, training_labels_2d)

    # Exercise 2: MED and GED Classifiers
    # Question 1: Implement MED Classifier
    print('Minimum Euclidean Distance Classifier: implemented in function med_classifier()')

    # Question 1: Implement GED Classifier
    print('General Euclidean Distance Classifier: implemented in function ged_classifier()')

    # Question 1: Determine decision boundaries for MED and GED
    w, wo = med_decision_boundary_coefficients(class_zero_prototype, class_one_prototype)
    print(f'The coefficients of the MED decision boundary are w = {w.T} and wo = {wo}')

    q0, q1, q2 = ged_decision_boundary_coefficients(
        class_zero_prototype, class_one_prototype, pca_20d_training_data, training_labels_20d
    )
    print(f'The coefficients of the GED decision boundary are q0 = {q0}, q1 = {q1}, and q2 = {q2}')

    # Question 2: Make label predictions for all test vectors and plot the prediction error for both classifiers
    med_20d_true_class_zero, med_20d_true_class_one, med_20d_false_class_zero, med_20d_false_class_one, med_20d_prediction_error = med_prediction_error(
        pca_20d_test_data, test_labels_20d, class_zero_prototype, class_one_prototype
    )
    ged_20d_true_class_zero, ged_20d_true_class_one, ged_20d_false_class_zero, ged_20d_false_class_one, ged_20d_prediction_error = ged_prediction_error(
        pca_20d_test_data,
        test_labels_20d,
        pca_20d_training_data,
        training_labels_20d,
        class_zero_prototype,
        class_one_prototype
    )

    # 20D MED/GED prediction error
    plot_med_ged_prediction_error(
        med_20d_false_class_zero,
        med_20d_false_class_one,
        med_20d_prediction_error,
        ged_20d_false_class_zero,
        ged_20d_false_class_one,
        ged_20d_prediction_error
    )

    # Question 4: Convert training images to 2x1 vectors and plot decision boundaries for MED and GED
    # Question 5: Find the confusion matrices for MED, GED, and kNN (k = [1, 5]) Classifiers using test datasets
    print('Minimum Euclidean Distance Classifier with 2x1 vectors: implemented in function med_classifier()')
    med_2d_true_class_zero, med_2d_true_class_one, med_2d_false_class_zero, med_2d_false_class_one, med_2d_prediction_error = med_prediction_error(
        pca_2d_test_data, test_labels_2d, class_zero_prototype_2d, class_one_prototype_2d
    )

    w_2d, wo_2d = med_decision_boundary_coefficients(class_zero_prototype_2d, class_one_prototype_2d)
    print(f'The coefficients of the 2D MED decision boundary are w = {w_2d.T} and wo = {wo_2d}')
    plot_2d_med_decision_boundary(pca_2d_training_data, training_labels_2d, w_2d, wo_2d)

    print('General Euclidean Distance Classifier with 2x1 vectors: implemented in function ged_classifier_2d()')
    ged_2d_true_class_zero, ged_2d_true_class_one, ged_2d_false_class_zero, ged_2d_false_class_one, ged_2d_prediction_error = ged_2d_prediction_error(
        pca_2d_test_data,
        test_labels_2d,
        pca_2d_training_data,
        training_labels_2d,
        class_zero_prototype_2d,
        class_one_prototype_2d
    )

    q0_2d, q1_2d, q2_2d = ged_decision_boundary_coefficients(
        class_zero_prototype_2d, class_one_prototype_2d, pca_2d_training_data, training_labels_2d
    )
    print(f'The coefficients of the GED decision boundary are q0 = {q0_2d}, q1 = {q1_2d}, and q2 = {q2_2d}')
    plot_2d_ged_decision_boundary(
        pca_2d_training_data, training_labels_2d, q0_2d, q1_2d, q2_2d
    )

    # 2D MED/GED prediction error
    plot_med_ged_prediction_error(
        med_2d_false_class_zero,
        med_2d_false_class_one,
        med_2d_prediction_error,
        ged_2d_false_class_zero,
        ged_2d_false_class_one,
        ged_2d_prediction_error
    )
