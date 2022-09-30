from GeneralEuclideanDistance import GeneralEuclideanDistance


def ged_classifier(training_dataset, training_labels, test_data_point, class_zero_prototype, class_one_prototype):
    ged = GeneralEuclideanDistance(training_dataset, training_labels, test_data_point)
    class_zero_ged = ged.class_zero_ged
    class_one_ged = ged.class_one_ged

    print('ged classifier')
    print(test_data_point.shape)

    if class_zero_ged < class_one_ged:
        return 0
    else:
        return 1
