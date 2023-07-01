import numpy as np


def counting_heuristic(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the total number of correctly classified instances for a given
    feature index, using the counting heuristic.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: int, total number of correctly classified instances
    """

    correct_count = 0

    feature_values = x_inputs[:, feature_index]

    for feature_value in np.unique(feature_values):
        idx = np.where(feature_values == feature_value)[0]
        sample_class_counts = np.array([np.sum(y_outputs[idx] == c) for c in classes])
        predicted_label = classes[np.argmax(sample_class_counts)]
        correct_count += np.sum(y_outputs[idx] == predicted_label)

    return correct_count


def set_entropy(x_inputs, y_outputs, classes):
    """Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """

    _, counts = np.unique(y_outputs, return_counts=True)
    probabilities = counts / len(y_outputs)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def information_remainder(x_inputs, y_outputs, feature_index, classes):
    """Calculate the information remainder after splitting the input-output set based on the
given feature index.


    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, information remainder value
    """

    # Get the values of the given feature
    original_entropy = set_entropy(x_inputs,y_outputs, classes)

    unique_feature_values = np.unique(x_inputs[:, feature_index])
    entropy_after_split = 0
    for value in unique_feature_values:
        mask = x_inputs[:, feature_index] == value
        entropy_after_split += np.sum(mask) / len(y_outputs) * set_entropy(x_inputs,y_outputs[mask], classes)

    gain = original_entropy - entropy_after_split
    return gain
