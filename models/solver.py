import sys

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import math

from models.ensembles.xgb import *
from models.lessers.knn import *
from models.lessers.lr import *
from models.neural_networks.cnn import *
from utils.dataset import *
from utils.logs import global_info
from utils.metrics import label_repartition
from utils.plots import *
from utils.submission import *


# TODO : Automatic init structural graph.
ML = {
    'supervised': {
        'classification': {
            "image"
        },
        'regression': {}
    },
    'unsupervised': {
        'clustering',
        'dimension_reduction'
    },
    'reinforcement': []
}


class KaggleCompSolver:
    def __init__(self,
                 comp_name='localhost',
                 comp_method_type='auto',  # supervised:classification
                 comp_task_type='auto',  # image
                 comp_algo='auto',  # cnn
                 train_label_pos=-1,
                 test_size=0.1,
                 data_path='data/',
                 train_path='train/',
                 test_path='test/',
                 sub_path='sub_samples/',
                 plots_path='docs/plots/',
                 preds_path='predictions/',
                 snapshots_path='model_snapshots/',
                 logs_path='logs/',
                 train_name_suffix='_train',
                 test_name_suffix='_test',
                 sub_name_suffix='_sub_sample'):
        # TODO : automatic init.
        if comp_method_type == 'auto' and comp_task_type == 'auto' and comp_algo == 'auto':
            raise ValueError('As for now, automatic initialization for comp method, task and algo is not supported.')

        # Global variables init.
        self.random_state = 42
        self.test_size = test_size

        self.comp_name = comp_name
        self.comp_method_type = comp_method_type
        self.comp_task_type = comp_task_type
        self.comp_algo = comp_algo
        self.full_name = comp_name + '-' + comp_method_type + '-' + comp_task_type + '-' + comp_algo + '_'
        self.data_path = data_path
        self.plots_path = plots_path
        self.preds_path = preds_path
        self.snapshots_path = snapshots_path
        self.logs_path = logs_path

        # Data init.
        train_file_path = train_path + comp_name + train_name_suffix
        test_file_path = test_path + comp_name + test_name_suffix
        sub_file_path = sub_path + comp_name + sub_name_suffix

        self.train, self.test, self.sub = datasets_from_files(train_file_path, test_file_path, sub_file_path, data_path)

        orig_X_train, orig_y_train = get_train_Xy(self.train, train_label_pos)

        self.orig_X_train = orig_X_train
        self.orig_y_train = orig_y_train

        self.X_train, self.X_test, self.y_train, self.y_test \
            = train_test_split(orig_X_train, orig_y_train, test_size=test_size, random_state=self.random_state)

    # ==================================================================================================================
    # Solver functions
    # ==================================================================================================================
    def info(self,
             logs=False):
        orig_stdout = sys.stdout
        full_pathname = self.data_path + self.logs_path + self.full_name + 'logs.txt'

        if logs:
            with open(full_pathname, 'w') as f:
                sys.stdout = f
                global_info(
                    self.X_train, self.X_test, self.y_train, self.y_test,
                    self.orig_X_train, self.orig_y_train, self.test, self.test_size
                )
                sys.stdout = orig_stdout
        else:
            global_info(
                self.X_train, self.X_test, self.y_train, self.y_test,
                self.orig_X_train, self.orig_y_train, self.test, self.test_size
            )

    def data_parse(self,
                   train_columns_drop=[],
                   test_columns_drop=[]):
        if len(train_columns_drop) > 0:
            self.orig_X_train = self.orig_X_train.drop(train_columns_drop, axis=1)
        else:
            print("No columns specified for train data, no changes made.")

        if len(test_columns_drop) > 0:
            self.test = self.test.drop(test_columns_drop, axis=1)
        else:
            print("No columns specified for test data, no changes made.")

    def solve(self,
              model=None,
              objective=None,
              metric_type='acc',
              epochs=50,
              batch_size=64):
        # TODO : automatic solving with multiple algorithms if type and/or algo set to 'auto'.

        # Manual selection algortihms.
        if self.comp_algo == 'all':
            self.knn(metric_type)
            self.lr(metric_type)
            self.xgb(objective, metric_type)

        if self.comp_algo == 'knn':
            self.knn(metric_type)

        if self.comp_algo == 'lr':
            self.lr(metric_type)

        if self.comp_algo == 'xgb':
            self.xgb(objective, metric_type)

        if self.comp_algo == 'cnn':
            self.cnn_image(model, batch_size, epochs)

    # ==================================================================================================================
    # Models
    # ==================================================================================================================
    def knn(self,
            metric_type):
        """
        K Nearest Neighbors

        :param metric_type:
        :return:
        """
        model = knn_train()
        y_pred, result = knn_predict(model, self.orig_X_train, self.orig_y_train, self.test, metric_type)

        self.save_result(y_pred, result)

    def lr(self,
           metric_type):
        """
        Logistic Regression

        :param metric_type:
        :return:
        """
        best_score, best_params = lr_objective(self.orig_X_train, self.orig_y_train, self.random_state)
        model = lr_train(best_params)
        y_pred, result = lr_predict(model, self.orig_X_train, self.orig_y_train, self.test, metric_type)

        self.save_result(y_pred, result)

    def cnn_image(self,
                  model,
                  batch_size=64,
                  epochs=50):
        """
        CNN towards vision/images

        :param model:
        :param batch_size:
        :param epochs:
        :return:
        """
        epochs_batches = str(epochs) + 'epochs-' + str(batch_size) + 'batch'
        full_pathname_model = self.data_path + self.snapshots_path + self.full_name + epochs_batches

        img_dim = int(math.sqrt(self.X_train.shape[1]))
        X_train = self.X_train.to_numpy().reshape((-1, img_dim, img_dim, 1))
        X_test = self.X_test.to_numpy().reshape((-1, img_dim, img_dim, 1))
        true_X_test = self.test.to_numpy().reshape((-1, img_dim, img_dim, 1))
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)

        X_train, X_test, true_X_test = norm_pixels(X_train, X_test, true_X_test)

        # CNN logic (neural_networks/cnn)
        accuracy, history = cnn_train(
            model=model,
            batch_size=batch_size,
            epochs=epochs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            full_pathname_model=full_pathname_model
        )

        self.plot_cnn_model(model)
        self.plot_curves(history, epochs_batches)
        self.plot_conf_matrix(model, epochs_batches, X_test, y_test)

        y_pred = cnn_predict(
            test=true_X_test,
            full_pathname_model=full_pathname_model,
        )

        self.save_result(y_pred, accuracy)

    def xgb(self,
            objective,
            metric_type):
        """
        XGBoost (Gradient Boosting)

        :param objective:
        :param metric_type:
        :return:
        """
        best_hyperparams = xgb_train(objective, self.X_train, self.y_train, self.X_test, self.y_test)
        y_pred, result = xgb_predict(best_hyperparams, self.orig_X_train, self.orig_y_train, self.test, metric_type)
        self.save_result(y_pred, result)

    # ==================================================================================================================
    # Plotting
    # ==================================================================================================================
    def plot_cnn_model(self, model):
        full_pathname = self.plots_path + self.full_name
        plot_keras_model(model, full_pathname)

    def plot_curves(self, history, epochs_batches):
        full_pathname = self.plots_path + self.full_name + epochs_batches + '_'
        plot_curves(history, full_pathname)

    def plot_conf_matrix(self, model, epochs_batches, X_test, y_test):
        full_pathname = self.plots_path + self.full_name + epochs_batches + '_'
        plot_conf_matrix(model, X_test, y_test, full_pathname)

    # ==================================================================================================================
    # Result saving
    # ==================================================================================================================
    def save_result(self, y_pred, accuracy):
        full_path = self.data_path + self.preds_path
        full_name = self.full_name + str(accuracy)
        save_data(self.sub, y_pred, full_path, full_name)
