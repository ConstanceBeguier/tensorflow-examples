#!/usr/bin/env python3
"""
TensorFlow code to learn the following network on the MNIST database
[28, 28, 1] -> Conv2D(30, (5, 5), relu) -> MaxPooling(2, 2) -> Conv2D(15, (3, 3), relu)
 -> MaxPooling(2, 2) -> Flatten -> Dense(128, relu) -> Dense(50, relu) -> Dense(10, softmax)
A SMAPE loss is defined.

Copyright (c) 2020 ConstanceMorel
Licensed under the MIT License
Written by Constance Beguier
"""

# Third-party library imports
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def smape_loss(target, prediction):
    """
    Define SMAPE loss
    """
    denominators = (tf.keras.backend.abs(target) + tf.keras.backend.abs(prediction))/2
    denominators_without_zero = tf.where(tf.equal(denominators, 0),
                                         tf.ones_like(denominators, dtype="float32"),
                                         denominators)
    numerators = tf.keras.backend.abs(target-prediction)
    return tf.keras.backend.mean(tf.math.divide(numerators, denominators_without_zero))

def main():
    """
    Learning the following network on the MNIST database
    [28, 28, 1] -> Conv2D(30, (5, 5), relu) -> MaxPooling(2, 2) -> Conv2D(15, (3, 3), relu)
    -> MaxPooling(2, 2) -> Flatten -> Dense(128, relu) -> Dense(50, relu) -> Dense(10, softmax)
    """
    # Load database
    (images_train, targets_train), (images_test, targets_test) = tf.keras.datasets.mnist.load_data()

    # Normalization
    images_train = images_train.reshape(-1, 784).astype(float)
    scaler = StandardScaler()
    images_train = scaler.fit_transform(images_train)
    images_test = images_test.reshape(-1, 784).astype(float)
    images_test = scaler.transform(images_test)

    images_train = images_train.reshape(-1, 28, 28, 1).astype(float)
    images_test = images_test.reshape(-1, 28, 28, 1).astype(float)

    # One hot encoding
    targets_train = tf.keras.utils.to_categorical(targets_train)
    targets_test = tf.keras.utils.to_categorical(targets_test)

    # Network architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), \
        activation="relu", padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3), activation="relu", padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # Optimizer
    model.compile(
        loss=smape_loss,
        optimizer="sgd",
        metrics=["accuracy"])

    # Learn
    history = model.fit(images_train, targets_train, epochs=10, validation_split=0.2)

    loss_curve = history.history["loss"]
    acc_curve = history.history["accuracy"]
    loss_val_curve = history.history["val_loss"]
    acc_val_curve = history.history["val_accuracy"]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(loss_curve, label="Train")
    plt.plot(loss_val_curve, label="Test")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(acc_curve, label="Train")
    plt.plot(acc_val_curve, label="Test")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("loss_acc_1.png")

    # Evaluate on the test database
    scores = model.evaluate(images_test, targets_test, verbose=0)
    print(scores)

if __name__ == '__main__':
    main()
