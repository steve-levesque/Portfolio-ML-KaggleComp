from sklearn.neighbors import KNeighborsClassifier
import warnings

from utils import predict

warnings.filterwarnings('ignore')


def knn_train():
    model = KNeighborsClassifier(n_neighbors=1)

    return model


def knn_predict(model, X_train, y_train, test, metric_type):
    y_pred, result = predict.get_prediction_with_metric(X_train, y_train, test, model, metric_type)

    return y_pred, result
