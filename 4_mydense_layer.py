#!/usr/bin/env python3
"""
TensorFlow code to learn the following network on the MNIST database
[28, 28, 1] -> Conv2D(30, (5, 5), relu) -> MaxPooling(2, 2) -> Conv2D(15, (3, 3), relu)
 -> MaxPooling(2, 2) -> Flatten -> Dense(128, relu) -> Dense(50, relu) -> Dense(10, softmax)
A dense layer has been created from tf.keras.layers.Layer.

Copyright (c) 2020 ConstanceMorel
Licensed under the MIT License
Written by Constance Beguier
"""

# Third-party library imports
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MyDense(tf.keras.layers.Layer):
    """
    Define MyDense layer
    """

    def __init__(self, output_dim, activation, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super(MyDense, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Create the weights of the layer
        """
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyDense, self).build(input_shape)

    def call(self, x):
        """
        Define the computations to perform in the layer
        """
        output = tf.keras.backend.dot(x, self.kernel)
        if self.activation == "relu":
            output = tf.nn.relu(output)
        elif self.activation == "sigmoid":
            output = tf.nn.sigmoid(output)
        elif self.activation == "softmax":
            output = tf.nn.softmax(output)
        return output

    def compute_output_shape(self, input_shape):
        """
        Define the output shape of the layer
        """
        return (input_shape[0], self.output_dim)

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
    # Network architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(30, (5, 5), input_shape=(28, 28, 1), \
        activation="relu", padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(15, (3, 3), activation="relu", padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(MyDense(output_dim=128, activation="relu"))
    model.add(MyDense(output_dim=50, activation="relu"))
    model.add(MyDense(output_dim=10, activation="softmax"))

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
    plt.savefig("loss_acc.png")

    # Evaluate on the test database
    scores = model.evaluate(images_test, targets_test, verbose=0)
    print(scores)

if __name__ == '__main__':
    main()
