import numpy as np
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator


def cnn_train(model, batch_size, epochs, X_train, y_train, X_test, y_test, full_pathname_model):
    gen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        horizontal_flip=False,
        vertical_flip=False
    )

    train_gen = gen.flow(X_train, y_train, batch_size=batch_size)
    test_gen = gen.flow(X_test, y_test, batch_size=batch_size)

    reduce_lr = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

    history = model.fit(train_gen,
                        epochs=epochs,
                        steps_per_epoch=X_train.shape[0] // batch_size,
                        validation_data=test_gen,
                        validation_steps=X_test.shape[0] // batch_size,
                        callbacks=[reduce_lr])

    model.save(full_pathname_model + '.h5')

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(acc * 100.0)

    return acc, history


def cnn_predict(test, full_pathname_model):
    model = load_model(full_pathname_model + '.h5')

    predictions = np.zeros(test.shape[0])

    counter = 0
    for img in test:
        img = np.expand_dims(img, axis=0)
        predict_value = model.predict(img)
        digit = np.argmax(predict_value)
        predictions[counter] = digit
        # print(counter)
        counter = counter + 1

    predictions = predictions.astype(int)

    return predictions
