#!/usr/bin/env python3
"""
TensorFlow code to learn the following network on the MNIST database
[28, 28, 1] -> Conv2D(30, (5, 5), relu) -> MaxPooling(2, 2) -> Conv2D(15, (3, 3), relu)
 -> MaxPooling(2, 2) -> Flatten -> Dense(128, relu) -> Dense(50, relu) -> Dense(10, softmax)
A gradient tape is used to redefine the learning step by step.

Copyright (c) 2020 ConstanceMorel
Licensed under the MIT License
Written by Constance Beguier
"""

# Third-party library imports
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

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

    # Learn
    optimizer = tf.keras.optimizers.SGD()

    @tf.function
    def train_step(images, targets):
        """
        Define the training step by step
        """
        # Save all operations
        with tf.GradientTape() as tape:
            # Make prediction
            predictions = model(images)
            # Compute loss
            loss = tf.keras.losses.categorical_crossentropy(targets, predictions)
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        # Update model
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    batch_size = 32
    epochs = 10
    images_per_epoch = len(images_train) // batch_size
    for _ in range(epochs):
        for i in range(images_per_epoch):
            start = i*batch_size
            train_step(images_train[start:start+batch_size], targets_train[start:start+batch_size])

    # Compile must be defined to use evaluate method
    model.compile(
        loss="categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"])

    # Evaluate on the test database
    scores = model.evaluate(images_test, targets_test, verbose=0)
    print(scores)

if __name__ == '__main__':
    main()
