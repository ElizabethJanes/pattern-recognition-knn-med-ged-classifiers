import os
import numpy as np
import torch
from torchvision import datasets
from sklearn.decomposition import PCA


def get_knn_training_dataset():
    project_dir = os.getcwd()
    training_data_dir = os.path.join(project_dir, 'training_dataset')
    training_dataset = datasets.MNIST(root=training_data_dir, train=True, download=True)
    training_data = training_dataset.data.numpy()
    training_labels = training_dataset.targets.numpy()

    target_indices = np.where((training_labels == 0) | (training_labels == 1))
    # print(target_indices)

    target_training_data = training_data[target_indices]
    target_training_labels = training_labels[target_indices]

    # print(target_training_data.shape)
    # print(target_training_labels.shape)

    reshaped_training_data = np.reshape(target_training_data, (len(target_training_data), 784))
    # print(reshaped_training_data.shape)

    pca_training_data = PCA(n_components=2)
    pca_reshaped_training_data = pca_training_data.fit_transform(reshaped_training_data)
    # print(pca_reshaped_training_data.shape)

    return pca_reshaped_training_data, target_training_labels


def get_med_ged_training_dataset():
    project_dir = os.getcwd()
    training_data_dir = os.path.join(project_dir, 'training_dataset')
    training_dataset = datasets.MNIST(root=training_data_dir, train=True, download=True)
    training_data = training_dataset.data.numpy()
    training_labels = training_dataset.targets.numpy()

    target_indices = np.where((training_labels == 0) | (training_labels == 1))
    print(target_indices)

    target_training_data = training_data[target_indices]
    target_training_labels = training_labels[target_indices]

    print(target_training_data.shape)
    print(target_training_labels.shape)

    reshaped_training_data = np.reshape(target_training_data, (len(target_training_data), 784))
    print(reshaped_training_data.shape)

    pca_training_data = PCA(n_components=20)
    pca_reshaped_training_data = pca_training_data.fit_transform(reshaped_training_data)
    print('20x1 pca data')
    print(pca_reshaped_training_data.shape)

    return pca_reshaped_training_data, target_training_labels


def get_knn_test_dataset():
    project_dir = os.getcwd()
    test_data_dir = os.path.join(project_dir, 'test_dataset')
    test_dataset = datasets.MNIST(root=test_data_dir, train=False, download=True)
    test_data = test_dataset.data.numpy()
    training_labels = test_dataset.targets.numpy()

    target_indices = np.where((training_labels == 0) | (training_labels == 1))
    # print(target_indices)

    target_test_data = test_data[target_indices]
    target_test_labels = training_labels[target_indices]

    # print(target_test_data.shape)
    # print(target_test_labels.shape)

    reshaped_test_data = np.reshape(target_test_data, (len(target_test_data), 784))
    # print(reshaped_test_data.shape)

    pca_test_data = PCA(n_components=2)
    pca_reshaped_test_data = pca_test_data.fit_transform(reshaped_test_data)
    # print(pca_reshaped_test_data.shape)

    return pca_reshaped_test_data, target_test_labels

def get_med_ged_test_dataset():
    project_dir = os.getcwd()
    test_data_dir = os.path.join(project_dir, 'test_dataset')
    test_dataset = datasets.MNIST(root=test_data_dir, train=False, download=True)
    test_data = test_dataset.data.numpy()
    training_labels = test_dataset.targets.numpy()

    target_indices = np.where((training_labels == 0) | (training_labels == 1))
    # print(target_indices)

    target_test_data = test_data[target_indices]
    target_test_labels = training_labels[target_indices]

    # print(target_test_data.shape)
    # print(target_test_labels.shape)

    reshaped_test_data = np.reshape(target_test_data, (len(target_test_data), 784))
    # print(reshaped_test_data.shape)

    pca_test_data = PCA(n_components=20)
    pca_reshaped_test_data = pca_test_data.fit_transform(reshaped_test_data)
    # print(pca_reshaped_test_data.shape)

    return pca_reshaped_test_data, target_test_labels
