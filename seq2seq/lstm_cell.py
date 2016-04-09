import tensorflow as tf
import numpy as np

class LSTMCell:
	def __init__(self, input_size, state_size, output_size):
		self.state_size = state_size
		self.input_size = input_size
		self.output_size = output_size

		with tf.variable_scope("LSTM"):
			w_ii = tf.get_variable("w_ii", [input_size, state_size])
			w_si = tf.get_variable("w_si", [state_size, state_size])
			w_hi = tf.get_variable("w_hi", [output_size, state_size])

			w_if = tf.get_variable("w_if", [input_size, state_size])
			w_sf = tf.get_variable("w_sf", [state_size, state_size])
			w_hf = tf.get_variable("w_hf", [state_size, state_size])

			w_io = tf.get_variable("w_io", [input_size, state_size])
			w_so = tf.get_variable("w_so", [state_size, state_size])
			w_ho = tf.get_variable("w_ho", [state_size, state_size])

			w_ic = tf.get_variable("w_ic", [input_size, state_size])
			w_hc = tf.get_variable("w_hc", [state_size, state_size])

	def zero_state(self, batch_size, dtype=tf.float64):
		zeros = tf.zeros([batch_size, state_size], dtype=dtype)

		return zeros

	def __call__(self, inputs, states, outputs):
		with tf.variable_scope("LSTM"):
			i = tf.tanh(tf.matmul(inputs, w_ii) + tf.matmul(states, w_si)\
					+  tf.matmul(outputs, w_hi))
			f = tf.tanh(tf.matmul(inputs, w_if) + tf.matmul(states, w_sf)\
					+  tf.matmul(outputs, w_hf))
			c = f * states + i * tf.sigmoid(tf.matmul(inputs, w_ic) + tf.matmul(outputs, w_hc))

			o = tf.tanh(tf.matmul(inputs, w_io) + tf.matmul(c, w_so)\
					+  tf.matmul(outputs, w_ho))

			new_outputs = o * tf.tanh(c)

		return c, new_outputs
