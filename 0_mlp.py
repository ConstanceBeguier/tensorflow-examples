#!/usr/bin/env python3
"""
TensorFlow code to learn the following network on the MNIST database
[784] -> Dense(300, relu) -> Dense(150, relu) -> Dense(10, softmax)

Copyright (c) 2020 ConstanceMorel
Licensed under the MIT License
Written by Constance Beguier
"""

# Third-party library imports
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def main():
    """
    Learning the following network on the MNIST database
    [784] -> Dense(300, relu) -> Dense(150, relu) -> Dense(10, softmax)
    """

    # Load database
    (images_train, targets_train), (images_test, targets_test) = tf.keras.datasets.mnist.load_data()

    # Normalization
    images_train = images_train.reshape(-1, 784).astype(float)
    scaler = StandardScaler()
    images_train = scaler.fit_transform(images_train)
    images_test = images_test.reshape(-1, 784).astype(float)
    images_test = scaler.transform(images_test)

    # Network architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(300, input_shape=[784], activation="relu"))
    model.add(tf.keras.layers.Dense(150, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    # model.summary()

    # Optimizer
    model.compile(
        loss="sparse_categorical_crossentropy",
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
    plt.savefig("loss_acc_curve.png")

    # Evaluate on the test database
    scores = model.evaluate(images_test, targets_test, verbose=0)
    print(scores)

if __name__ == '__main__':
    main()
