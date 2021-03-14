"""
Michael Patel
March 2021

Project description:
    To classify NBA team logos

File description:

"""
################################################################################
# Imports
import os
import matplotlib.pyplot as plt
import tensorflow as tf


################################################################################
# directories
DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
SAVE_DIR = os.path.join(os.getcwd(), "saved_model")

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3

NUM_EPOCHS = 50000
BATCH_SIZE = 30
