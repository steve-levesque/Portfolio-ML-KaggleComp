from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense

import models.solver


def define_model():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


if __name__ == '__main__':
    print("Start of example usage with KaggleCompSolver for mnist ...")

    cnn_model = define_model()

    solver = models.solver.KaggleCompSolver(
        comp_name='mnist',
        train_label_pos=0,
        comp_method_type='supervised-classification',
        comp_task_type='image',
        comp_algo='cnn'
    )

    solver.info(logs=True)

    # solver.cnn_image(cnn_model, batch_size=128, epochs=2)

    solver.solve(model=cnn_model, objective='multi:softmax', metric_type='acc', batch_size=128, epochs=2)
