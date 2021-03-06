#!/usr/bin/env python3
"""
TensorFlow code to learn the following network on the MNIST database
Concat([output(Network1), output(Network2)]) -> Dense(50, relu) -> Dense(10, softmax)
Network1
[28, 28, 1] -> Conv2D(30, (5, 5), relu) -> MaxPooling(2, 2) -> Conv2D(15, (3, 3), relu)
 -> MaxPooling(2, 2) -> Flatten -> Dense(128, relu) -> Dense(50, relu)
Network2
[784] -> Dense(300, relu) -> Dense(150, relu) -> Dense(50, relu)

Copyright (c) 2020 ConstanceMorel
Licensed under the MIT License
Written by Constance Beguier
"""

# Third-party library imports
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Network architecture
# pylint: disable=R0902,R0903
class NonSequentialModel(tf.keras.Model):
    """
    Define a non-sequential model
    """
    def __init__(self):
        super(NonSequentialModel, self).__init__()
        self.network1_layer1 = tf.keras.layers.Conv2D(30, (5, 5), activation="relu", padding='same')
        self.network1_layer2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.network1_layer3 = tf.keras.layers.Conv2D(15, (3, 3), activation="relu", padding='same')
        self.network1_layer4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.network1_layer5 = tf.keras.layers.Flatten()
        self.network1_layer6 = tf.keras.layers.Dense(128, activation="relu")
        self.network1_layer7 = tf.keras.layers.Dense(50, activation="relu")

        self.network2_layer1 = tf.keras.layers.Flatten()
        self.network2_layer2 = tf.keras.layers.Dense(300, activation="relu")
        self.network2_layer3 = tf.keras.layers.Dense(150, activation="relu")
        self.network2_layer4 = tf.keras.layers.Dense(50, activation="relu")

        self.network_layer1 = tf.keras.layers.Concatenate()
        self.network_layer2 = tf.keras.layers.Dense(50, activation="relu")
        self.network_layer3 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        """
        Define how the different layers are linked together
        """
        n1_l1 = self.network1_layer1(x)
        n1_l2 = self.network1_layer2(n1_l1)
        n1_l3 = self.network1_layer3(n1_l2)
        n1_l4 = self.network1_layer4(n1_l3)
        n1_l5 = self.network1_layer5(n1_l4)
        n1_l6 = self.network1_layer6(n1_l5)
        n1_l7 = self.network1_layer7(n1_l6)

        n2_l1 = self.network2_layer1(x)
        n2_l2 = self.network2_layer2(n2_l1)
        n2_l3 = self.network2_layer3(n2_l2)
        n2_l4 = self.network2_layer4(n2_l3)

        n_l1 = self.network_layer1([n1_l7, n2_l4])
        n_l2 = self.network_layer2(n_l1)
        n_l3 = self.network_layer3(n_l2)
        return n_l3

def main():
    """
    Learning the following network on the MNIST database
    Concat([output(Network1), output(Network2)]) -> Dense(50, relu) -> Dense(10, softmax)
    Network1
    [28, 28, 1] -> Conv2D(30, (5, 5), relu) -> MaxPooling(2, 2) -> Conv2D(15, (3, 3), relu)
    -> MaxPooling(2, 2) -> Flatten -> Dense(128, relu) -> Dense(50, relu)
    Network2
    [784] -> Dense(300, relu) -> Dense(150, relu) -> Dense(50, relu)
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


    model = NonSequentialModel()

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
