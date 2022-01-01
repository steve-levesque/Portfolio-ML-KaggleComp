import pandas as pd


def datasets_from_files(train_file, test_file, sub_file, path='../../data/', ext='.csv'):
    train = pd.read_csv(path + train_file + ext)
    test = pd.read_csv(path + test_file + ext)
    submission = pd.read_csv(path + sub_file + ext)

    return train, test, submission


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_train_Xy(train, train_label_pos):
    n_features = train.shape[1]
    if train_label_pos == -1:
        pos = 1
    else:
        pos = 0
    X_train = train.iloc[:, 1:n_features - pos]
    y_train = train.iloc[:, train_label_pos]
    return X_train, y_train


def norm_pixels(train, test, true_test, spectrum=255.0):
    train_norm = train.astype('float32') / spectrum
    test_norm = test.astype('float32') / spectrum
    true_test_norm = true_test.astype('float32') / spectrum

    return train_norm, test_norm, true_test_norm
