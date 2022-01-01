from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import warnings

from utils import predict

warnings.filterwarnings('ignore')


def lr_objective(X_train, y_train, random_state):
    model = LogisticRegression()
    solvers = ['newton-cg']
    penalty = ['l2']
    c_values = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]  # [100, 25, 10, 3, 1.0, 0.1, 0.01]

    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=random_state)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(X_train, y_train)

    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']

    best_score = grid_result.best_score_
    best_params = grid_result.best_params_

    print("grid_result.best_score_: " + str(best_score))
    print("grid_result.best_params_: " + str(best_params))

    return best_score, best_params


def lr_train(best_params):
    params_dict = best_params
    params_dict['max_iter'] = 500
    model = LogisticRegression(**params_dict)

    return model


def lr_predict(model, X_train, y_train, test, metric_typ):
    y_pred, result = predict.get_prediction_with_metric(X_train, y_train, test, model, metric_typ)

    return y_pred, result
