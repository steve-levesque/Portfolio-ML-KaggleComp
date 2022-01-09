from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


def sc_run(models, X_train, y_train, test):
    model = StackingClassifier(
        estimators=models,
        final_estimator=LogisticRegression(C=0.1,
                                           multi_class='multinomial',
                                           solver='lbfgs')
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(test)

    return y_pred
