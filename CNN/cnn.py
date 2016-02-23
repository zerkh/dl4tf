import tensorflow as tf
import numpy as np
import os
from input_data import get_training_data

tf.app.flags.DEFINE_integer("layer", 1, "amount of layers in CNN")
tf.app.flags.DEFINE_string("data_dir", "./data", "the dir to store data")
tf.app.flags.DEFINE_float("learning_rate", "0.01", "learning rate")
tf.app.flags.DEFINE_integer("max_iter", "50", "max times to do iteration")

FLAGS = tf.app.flags.FLAGS

def training(sess, train_X, train_Y):
	max_iter = FLAGS.max_iter
	lr = FLAGS.learning_rate

	X_ = train_X.reshape(train_X.shape[0], 28,28, 1)
	Y_ = np.zeros((len(train_Y), 10), dtype=np.float32)

	for i in range(0, len(train_Y)):
		Y_[i][train_Y[i]] = 1.0

	X = tf.placeholder("float32", shape=[None, None, None, None], name="X")
	Y = tf.placeholder("float32", shape=[None, None], name="Y")

	#conv1
	conv1_kernel = tf.Variable(np.random.uniform(0.0, 1.0, size=[1, 10, 1, 3]), name="conv1_kernel")

	conv1 = tf.nn.conv2d(X, conv1_kernel, [1,1,1,1], padding="SAME")

	#pool = tf.nn.avg_pool(conv1, [1,1,1,3], [1,1,1,1], padding="SAME")

	sess.run(tf.initialize_all_variables())

	reshape = tf.reshape(conv1, [train_X.shape[0], 30])

	loss = tf.nn.softmax_cross_entropy_with_logits(Y, reshape)

	opt = tf.train.AdagradOptimizer(lr)

	train_step = opt.minimize(loss)

	for step in max_iter:
		print "Iter %d" %(step)

		sess.run(train_step, feed_dict={X: X_, Y: Y_})

		print "Loss: %g" %(loss.eval(feed_dict={X: X_, Y: Y_}))

def main(_):
	train_X, train_Y = get_training_data("data/train.csv")

	with tf.Session() as sess:
		training(sess, train_X, train_Y)

		#pred_Y = test(sess, test_X)

if __name__ == "__main__":
	tf.app.run()
