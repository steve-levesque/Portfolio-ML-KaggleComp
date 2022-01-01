from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from functools import partial
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import warnings

from utils import dataset, metrics, predict, submission

warnings.filterwarnings('ignore')


# https://medium.com/analytics-vidhya/hyperparameter-tuning-hyperopt-bayesian-optimization-for-xgboost-and-neural-network-8aedf278a1c9
def xgb_objective(space, evaluation):
    clf = XGBClassifier(
        num_class=10,
        objective=space['objective'],
        n_estimators=space['n_estimators'],
        max_depth=int(space['max_depth']),
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree'])
    )

    X_train = evaluation[0][0]
    y_train = evaluation[0][1]
    X_test = evaluation[1][0]
    y_test = evaluation[1][1]

    clf.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        eval_metric="merror",
        early_stopping_rounds=10,
        verbose=False
    )

    pred = clf.predict(X_test)
    # pred_prob = clf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, pred)
    # loss = log_loss(y_test, pred_prob)
    # print("SCORE:", accuracy, loss)
    return {'loss': -accuracy, 'status': STATUS_OK}


def xgb_train(objective, X_train, y_train, X_test, y_test):
    trials = Trials()

    space = {'objective': objective,  # 'binary:logistic', 'multi:softmax'
             'max_depth': hp.choice("max_depth", np.arange(1, 18, dtype=int)),
             'learning_rate': hp.uniform('learning_rate', 0.01, 1),
             'gamma': hp.uniform('gamma', 1, 9),
             'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
             'reg_lambda': hp.uniform('reg_lambda', 0, 1),
             'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1),
             'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
             'n_estimators': 180,  # hp.choice("n_estimators", np.arange(1, 300, dtype=int)),
             'seed': 0
             }

    evaluation = [(X_train, y_train), (X_test, y_test)]

    xgb_objective_with_data = partial(xgb_objective, evaluation=evaluation)

    best_params = fmin(fn=xgb_objective_with_data,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=2,
                       trials=trials)

    print("The best hyperparameters are : ", "\n")
    best_params['objective'] = objective  # 'binary:logistic'
    print(best_params)

    return best_params


def xgb_predict(best_params, orig_X_train, orig_y_train, test, metric_type):
    xgbc = XGBClassifier(**best_params)
    y_pred = predict.get_prediction_with_metric(orig_X_train, orig_y_train, test, xgbc, metric_type)

    return y_pred
