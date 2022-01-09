import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# Aux. functions
def softmax(product):
    if len(product.shape) > 1:
        max_each_row = np.max(product, axis=1, keepdims=True)
        exps = np.exp(product - max_each_row)
        sum_exps = np.sum(exps, axis=1, keepdims=True)
        res = exps / sum_exps
    else:
        product_max = np.max(product)
        product = product - product_max
        numerator = np.exp(product)
        denominator = 1.0 / np.sum(numerator)
        res = numerator.dot(denominator)
    return res


def gradient_descent(X, y_onehot, theta, lambda_, eps, alpha, max_iter):
    losses = []
    i = 0
    print("Iteration: Cost")

    while (i < max_iter):
        i += 1
        grad = reg_gradient_softmax(X, y_onehot, theta, lambda_)
        theta -= alpha * grad

        loss = reg_cost_softmax(X, y_onehot, theta, lambda_)
        if (i % 500 == 0):
            print("{}: {:.8f}".format(i, loss))

        len_losses = len(losses)
        if (len_losses == 0):
            print("{}: {:.8f}".format(i, loss))
            diff = np.abs(loss)
        else:
            diff = np.abs(losses[len_losses - 1] - loss)

        losses.append(loss)
        if (diff < eps):
            return theta, losses

    return theta, losses


def reg_gradient_softmax(X, y_onehot, theta, lambda_):
    n_samples = X.shape[0]
    softmax_res = softmax(np.dot(X, theta.T))

    gradient = (-1.0 / n_samples) * np.dot((y_onehot - softmax_res).T, X)

    theta_without_bias = theta[:, 1:theta.shape[1]]
    reg = -lambda_ / n_samples * theta_without_bias

    gradient[:, 1:gradient.shape[1]] = gradient[:, 1:gradient.shape[1]] + reg

    return gradient


def reg_cost_softmax(X, y_onehot, theta, lambda_):
    n_samples = X.shape[0]
    softmax_res = softmax(np.dot(X, theta.T))  # (n_samples, n_classes)
    cost = - (1.0 / n_samples) * np.sum(y_onehot * np.log(softmax_res))

    theta_without_bias = theta[:, 1:theta.shape[1]]
    reg = lambda_ / n_samples * np.sum(theta_without_bias ** 2)
    return cost + reg


def iterative(X_train, y_train, theta0, lambda_, eps, alpha, max_iter):
    theta, losses = gradient_descent(X_train, y_train, theta0, lambda_, eps, alpha, max_iter)
    return theta, losses


def mat_prob_test(test_data, final_theta):
    res = test_data.dot(np.transpose(final_theta))
    return np.argmax(res.to_numpy(), axis=1)


def onehot_y(labels, classes):
    size = labels.shape[0]
    result = np.zeros((size, classes))
    for i in range(size):
        cl = int(labels[i])
        result[i][cl] = 1
    return result


def accuracy_percent(X_test, y_test, theta):
    X_test_array = X_test.to_numpy()
    mat = X_test_array.dot(theta)
    y_pred = np.argmax(mat, axis=1)
    y_test_array = y_test.to_numpy()
    accuracy_rate = np.sum(y_test_array == y_pred) / y_test_array.shape[0]
    return accuracy_rate


def preprocessing(features):
    X = features.copy()
    # print(X)
    X.insert(0, 'bias', 1)
    X_means = np.mean(X)
    X_std = np.std(X)
    X_scale = (X - X_means) / X_std
    X_scale.iloc[:, 0] = np.ones((X_scale.shape[0], 1))
    return X_scale


def hyperparameter_tuning(lambda_list, X_train, y_onehot, X_test, y_test, eps, alpha, max_iter, nb_classes):
    n = X_train.shape[1]
    all_theta = {}
    all_losses = {}
    print("Hyperparameter tuning: Lambda")
    for each_lambda in lambda_list:
        theta0 = np.zeros((n, nb_classes))
        print(each_lambda)
        theta, loss_dict = iterative(X_train, y_onehot, theta0, each_lambda, eps, alpha, max_iter, nb_classes)
        all_theta[each_lambda] = theta
        all_losses[each_lambda] = loss_dict
        accuracy = accuracy_percent(X_test, y_test, theta)
        print("accuracy for lambda = {}: {:.8f}".format(each_lambda, accuracy))
        print("-------------------------------------------------")

    return all_theta, all_losses


# Primary functions
def lr_scratch_objective(tmp):
    pass


def lr_scratch_train(tmp):
    pass


def lr_scratch_predict(tmp):
    pass


# Full run
def lr_scratch_run(X_train,
                   y_train,
                   test,
                   lambda_=3,
                   eps=10 ** -12,
                   alpha=0.85,
                   max_iter=4000):

    nb_classes = len(set(y_train))

    X_train = preprocessing(X_train)
    test = preprocessing(test)
    y_onehot = onehot_y(pd.Series.to_numpy(y_train.copy()), nb_classes)
    theta = np.zeros((nb_classes, X_train.shape[1]))

    final_theta, loss_dict_final = iterative(X_train, y_onehot, theta, lambda_, eps, alpha, max_iter)

    y_pred = mat_prob_test(test, final_theta)
    accuracy = accuracy_percent(X_train, y_train, np.transpose(final_theta))

    return y_pred, accuracy
