"""
Michael Patel
March 2021

Project description:
    To classify NBA team logos

File description:

"""
################################################################################
# Imports
from parameters import *


################################################################################
# MobileNetV2
def build_cnn_mobilenet(num_classes):
    mobilenet = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS),
        include_top=False,
        weights='imagenet')

    mobilenet.trainable = False

    # add classification head
    model = tf.keras.Sequential()
    model.add(mobilenet)
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation="relu"
    ))
    model.add(tf.keras.layers.Dropout(
        rate=0.2
    ))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(
        units=num_classes,
        activation="softmax"
    ))

    return model
