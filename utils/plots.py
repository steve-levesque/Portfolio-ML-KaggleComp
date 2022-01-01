import numpy as np
import seaborn as sns
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_keras_model(model, path):
    full_path = path + 'keras-model.png'
    plot_model(model, to_file=full_path, show_shapes=True, show_layer_names=True)


def plot_curves(history, full_pathname):
    fig, ax = plt.subplots(2, 1, figsize=(18, 10))
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss", axes=ax[0])
    ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    plt.savefig(full_pathname + 'curves.png')


def plot_conf_matrix(model, X_test, y_test, full_pathname):
    plt.figure(figsize=(10, 10))

    y_pred = model.predict(X_test)

    Y_pred = np.argmax(y_pred, 1)
    Y_test = np.argmax(y_test, 1)

    mat = confusion_matrix(Y_test, Y_pred)

    sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')

    plt.savefig(full_pathname + 'confusion-matrix.png')
