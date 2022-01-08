import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report


def get_accuracy(labels, y_pred):
    result = accuracy_score(labels, y_pred)
    return result


def get_f1(labels, y_pred):
    result = f1_score(labels, y_pred)
    return result


def get_classification_report(labels, y_pred):
    print(classification_report(labels, y_pred))


def label_repartition(labels):
    unique_set = set(labels)
    unique_list = list(unique_set)
    total = len(labels)
    length = len(unique_list)

    label_count = np.zeros(length)
    label_percents = np.zeros(length)

    for label in labels:
        label = int(label)
        label_count[label] = label_count[label] + 1

    for i in range(0, length):
        label_percents[i] = label_count[i] / total

    return label_count, label_percents
