import os
import tensorflow as tf
# from tensorflow.contrib.learn.python.learn.datasets import mnist

save_dir = "mnist"

data_sets = tf.keras.datasets.mnist.read_data_sets(save_dir,
                                 dtype=tf.uint8,
                                 reshape=False,
                                 validation_size=1000)
