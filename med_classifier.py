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


def med_decision_boundary(class_zero_prototype, class_one_prototype):
    # x = np.ndarray([1, 20])
    w = class_zero_prototype - class_one_prototype
    wo = 1/2*(np.dot(class_one_prototype, class_one_prototype) - np.dot(class_zero_prototype, class_zero_prototype))
    print('med decision boundary')
    print(w.shape)
    print(wo)
    # g = np.matmul((w.T), x) + wo
    x = np.linalg.solve(w, wo)

    # plt.plot(x, g)
    # plt.show()

