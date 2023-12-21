import numpy as np
import tensorflow as tf
import tensorflow.keras.layers
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout

def buah():
    train_dir = '/archive/train/'
    valid_dir = '/archive/validation/'

    training_set = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    validation_set = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    model = tf.keras.models.Sequential([
        Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[224, 224, 3]),
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dense(units=512, activation='relu'),
        Dense(units=512, activation='relu'),
        Dropout(0.5),
        Dense(units=36, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.001),
        metrics=["accuracy"]
    )

    model.fit(
        training_set,
        validation_data=validation_set,
        epochs=32,
    )

if __name__ == '__main__':
    model=buah()
    model.save("buah.h5")