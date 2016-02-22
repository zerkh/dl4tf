import tensorflow as tf
import numpy as np
import os

tf.app.flags.DEFINE_integer("layer", 1, "amount of layers in CNN")

FLAGS = tf.app.flags.FLAGS

def main(_):
	print FLAGS.layer

if __name__ == "__main__":
	tf.app.run()
