from GeneralEuclideanDistance import GeneralEuclideanDistance


def ged_classifier(training_dataset, training_labels, test_data_point, class_zero_prototype, class_one_prototype):
    class_zero_euclidean_dist = GeneralEuclideanDistance(
        training_dataset, training_labels, class_zero_prototype, 0, test_data_point
    )
    class_one_euclidean_dist = GeneralEuclideanDistance(
        training_dataset, training_labels, class_one_prototype, 1, test_data_point
    )

    print('ged classifier')
    print(test_data_point.shape)

    if class_zero_euclidean_dist.euclidean_distance < class_one_euclidean_dist.euclidean_distance:
        return 0
    else:
        return 1
