import numpy as np
import tensorflow as tf

class Node:
	def __init__(self, vec, l_n, r_n):
		self.vec = vec
		self.l_n = l_n
		self.r_n = r_n

class RAEModel:
	def __init__(self, layer_size=l_size):
		self.layer_size = layer_size

	def create_model(self, sess):
		l_size = self.layer_size

		self.W1 = tf.Variable(tf.truncated_normal([l_size, l_size], stddev=0.1), name="W1")
		self.W2 = tf.Variable(tf.truncated_normal([l_size, l_size], stddev=0.1), name="W2")
		self.b1 = tf.Variable(tf.constant(0.1, [l_size]), name="b1")
		self.b2 = tf.Variable(tf.constant(0.1, [l_size]), name="b2")

		self.r_W1 = tf.Variable(tf.truncated_normal([l_size, l_size], stddev=0.1), name="r_W1")
		self.r_W2 = tf.Variable(tf.truncated_normal([l_size, l_size], stddev=0.1), name="r_W2")
		self.r_b1 = tf.Variable(tf.constant(0.1, [l_size]), name="r_b1")
		self.r_b2 = tf.Variable(tf.constant(0.1, [l_size]), name="r_b2")

		sess.run(tf.initialize_all_variables())

	def get_loss(self, sess, l_n, r_n):
		hidden = tf.tanh(tf.matmul(l_n,self.W1)+tf.b1 + tf.matmul(r_n,self.W2)+tf.b2)
		r_l_n = tf.matmul(hidden,tf.r_W1)+self.r_b1
		r_r_n = tf.matmul(hidden,tf.r_W2)+self.r_b2
		loss = tf.sqrt((r_l_n-l_n)**2 + (r_r_n-r_n)**2)
		return loss
		

	#Get top node of the tree
	def build(self, sess, inputs):
		max_pos = 0
		max_loss = 0.0

		for pos in xrange(len(inputs)):
			loss = get_loss(sess, inputs[pos], inputs[pos+1])
			if max_loss < loss:
				max_loss = loss
				max_pos = pos

		
