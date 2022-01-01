import numpy as np
from utils import metrics


def get_prediction_with_metric(X_train, y_train, X_test, model, metric_type='acc'):
    result = 'unavailable'
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    if metric_type == 'acc':
        result = metrics.get_accuracy(y_train, y_pred_train)
        print('acc on training dataset: ')
        print(result)
    if metric_type == 'f1':
        result = metrics.get_f1(y_train, y_pred_train)
        print('f1_score on training dataset: ')
        print(result)

    metrics.get_classification_report(y_train, y_pred_train)
    y_pred = model.predict(X_test)
    y_pred = y_pred.astype(np.uint8)
    return y_pred, result
