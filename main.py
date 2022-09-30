from dataset_utilities import get_knn_training_dataset, get_med_ged_training_dataset, get_knn_test_dataset, get_med_ged_test_dataset
from knn_classifier import knn_classifier
from plot import plot_knn, plot_knn_decision_boundaries
from prediction_error import plot_knn_prediction_error, plot_med_prediction_error
from med_classifier import med_classifier, med_decision_boundary_coefficients
from classifier_utilities import get_class_prototypes
from ged_classifier import ged_classifier

if __name__ == '__main__':

    # Exercise 1: Nearest Neighbour Classifier
    # Question 1: Implement kNN Classifier, compute kNN solution for k = [1, 5]
    knn_pca_training_data, knn_training_labels = get_knn_training_dataset()
    knn_pca_test_data, knn_test_labels = get_knn_test_dataset()
    print(knn_pca_test_data[1].shape)
    print('k-Nearest Neighbour Classifier')
    for k in range(1, 6):
        knn_test_classification = knn_classifier(
            k, knn_pca_training_data, knn_training_labels, knn_pca_test_data[1]
        )
        print(f'When k = {k}, the test point is classified as {knn_test_classification} and the expected '
              f'classification is {knn_test_labels[1]}')

    # Question 1: Plot the classification boundaries between the two classes for k = [1, 5]
    # plot_knn(knn_pca_training_data, knn_training_labels)
    # plot_knn_decision_boundaries(knn_pca_training_data, knn_training_labels)

    # Question 2: Make label predictions for all test vectors and find the prediction error for the kNN Classifier
    # plot_knn_prediction_error(knn_pca_test_data, knn_test_labels, knn_pca_training_data, knn_training_labels)

    # Exercise 2: MED and GED Classifiers
    med_ged_pca_training_data, med_ged_training_labels = get_med_ged_training_dataset()
    med_ged_pca_test_data, med_ged_test_labels = get_med_ged_test_dataset()
    class_zero_prototype, class_one_prototype = get_class_prototypes(med_ged_pca_training_data, med_ged_training_labels)

    # Question 1: Implement MED Classifier
    med_test_classification = med_classifier(med_ged_pca_test_data[0], class_zero_prototype, class_one_prototype)
    print('Minimum Euclidean Distance Classifier')
    print(f'The test point is classified as {med_test_classification} and the expected classification '
          f'is {med_ged_test_labels[0]}')

    # Question 1: Implement GED Classifier
    ged_classifier(
        med_ged_pca_training_data,
        med_ged_training_labels,
        med_ged_pca_test_data[0],
        class_zero_prototype,
        class_one_prototype
    )

    # Question 1: Determine decision boundaries for MED and GED
    # plot_med_prediction_error(
    #     med_ged_pca_test_data, med_ged_test_labels, med_ged_pca_training_data, med_ged_training_labels
    # )
    w, wo = med_decision_boundary_coefficients(class_zero_prototype, class_one_prototype)
    print(f'The coefficients of the MED decision boundary are w = {w.T} and wo = {wo}')

    # Question 2: Make label predictions for all test vectors and plot the prediction error for both classifiers
    # (two separate plots)

    # Question 4: Convert training images to 2x1 vectors and plot decision boundaries for MED and GED

    # Question 5: Find the confusion matrices for MED, GED, and kNN (k = [1, 5]) Classifiers using test datasets

