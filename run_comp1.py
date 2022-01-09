import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from models.solver import KaggleCompSolver


# ======================================================================================================================
# Functions for data augmentation.
def month(times):
    month = []
    for each_time in times:
        time_str = str(each_time)
        each_month = time_str[4:6]
        month.append(int(each_month))
    return month


def add_seasons(data, orig_data):
    new_data = data.copy()
    month_list = month(orig_data['time'])
    new_data['winter'] = [1 if (x == 1 or x == 2 or x == 12) else 0 for x in month_list]
    new_data['summer'] = [1 if (x == 6 or x == 7 or x == 8) else 0 for x in month_list]
    new_data['automn'] = [1 if (x == 9 or x == 10 or x == 11) else 0 for x in month_list]
    new_data['spring'] = [1 if (x == 3 or x == 4 or x == 5) else 0 for x in month_list]

    return new_data


def features_scaling(features):
    X = features.copy()
    X_means = np.mean(X)
    X_std = np.std(X)
    X_scaled = (X - X_means) / X_std

    return X_scaled
# ======================================================================================================================


# ======================================================================================================================
# Models for stacking classifier
models = [
    ('xgb', XGBClassifier(
        objective='multi:softprob',
        eval_metric='merror',
        colsample_bytree=1,
        learning_rate=0.02,
        max_depth=4,
        n_estimators=10,
    )),
    ('gb', GradientBoostingClassifier()),
    ('rf', RandomForestClassifier(max_depth=8))
]
# ======================================================================================================================


if __name__ == '__main__':
    # ==================================
    # Logistic Regression from scratch
    # ==================================
    solver1 = KaggleCompSolver(
        comp_name='comp1',
        train_label_pos=-1,
        comp_method_type='supervised-classification',
        comp_task_type='data',
        comp_algo='lr_scratch'
    )

    solver1.info()
    columns_drop = ['S.No', 'PS', 'PRECT', 'time']
    solver1.data_parse(train_columns_drop=columns_drop, test_columns_drop=columns_drop)
    # solver2.data_parse(train_columns_drop=['S.No'], test_columns_drop=['S.No'])
    solver1.info(logs=True)

    solver1.solve()

    # ==================================
    # XGBoost
    # ==================================
    solver2 = KaggleCompSolver(
        comp_name='comp1',
        train_label_pos=-1,
        comp_method_type='supervised-classification',
        comp_task_type='data',
        comp_algo='xgb'
    )

    solver2.info()
    columns_drop = ['S.No']
    solver2.data_parse(train_columns_drop=columns_drop, test_columns_drop=columns_drop)
    solver2.info(logs=True)

    solver2.solve(objective='multi:softmax', metric_type='acc')

    # ==================================
    # Stacking Classifier
    # ==================================
    solver3 = KaggleCompSolver(
        comp_name='comp1',
        train_label_pos=-1,
        comp_method_type='supervised-classification',
        comp_task_type='data',
        comp_algo='sc'
    )
    solver3.info()

    # Data augmentation.
    orig_X_train, test = solver3.data_get_before_augmentation()
    columns_drop = ['S.No', 'PS', 'PRECT', 'time']
    solver3.data_parse(train_columns_drop=columns_drop, test_columns_drop=columns_drop)
    new_X_train, new_test = solver3.data_get_before_augmentation()

    new_X_train = features_scaling(new_X_train)
    new_test = features_scaling(new_test)
    X_train_add_season = add_seasons(new_X_train, orig_X_train)
    test_add_season = add_seasons(new_test, test)

    solver3.data_set_after_augmentation(X_train_add_season, test_add_season)

    # Log info and solve.
    solver3.info(logs=True)
    solver3.solve(model=models)
