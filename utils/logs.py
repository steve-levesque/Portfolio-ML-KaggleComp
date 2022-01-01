from utils.metrics import label_repartition


def global_info(X_train, X_test, y_train, y_test, orig_X_train, orig_y_train, test, test_size):
    print("=========================================================================")
    print("Train/Test split on original train percent: " + str(test_size * 100) + "%")
    print("X_train dim: " + str(X_train.shape))
    print("X_test dim: " + str(X_test.shape))
    print("y_train dim: " + str(y_train.shape))
    print("y_test dim: " + str(y_test.shape))
    print()
    print("orig_X_train dim: " + str(orig_X_train.shape))
    print("orig_y_train dim: " + str(orig_y_train.shape))
    print("test dim: " + str(test.shape))
    print()
    print("Label repartition: ")
    print(label_repartition(orig_y_train))
    print()
    print("Train and test merge or test subset of train biases: " + str(False))
    print("=========================================================================")