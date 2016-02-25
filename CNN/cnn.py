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

	X = tf.placeholder("float32", shape=[42000, 28, 28, 1], name="X")
	Y = tf.placeholder("float32", shape=[42000, 10], name="Y")

	#conv1
	conv1_kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 3], stddev=0.1), name="conv1_kernel", trainable=True)

	_ = tf.histogram_summary("conv1_kernel", conv1_kernel)

	conv1 = tf.nn.conv2d(X, conv1_kernel, [1,1,1,1], padding="SAME")

	b_conv1 = tf.Variable(tf.constant(0.1, shape=[3]))

	_ = tf.histogram_summary("b_conv1", b_conv1)

	relu = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))

	pool = tf.nn.avg_pool(relu, [1,2,2,1], [1,2,2,1], padding="SAME")

	pool_shape = pool.get_shape().as_list()

	reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

	#Weight
	W = tf.Variable(np.random.uniform(0.0, 1.0, size=[14*14*3, 10]).astype(np.float32), name="W", trainable=True)
	b = tf.Variable(np.zeros([10]).astype(np.float32), name="b")

	_ = tf.histogram_summary("W", W)
	_ = tf.histogram_summary("b", b)

	logits = tf.nn.softmax(tf.nn.bias_add(tf.matmul(reshape, W), b))

	pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
	acc = tf.reduce_mean(tf.cast(pred, tf.float32))

	loss = tf.reduce_mean(-Y*tf.log(logits))

	opt = tf.train.GradientDescentOptimizer(lr)

	train_step = opt.minimize(loss)

	summary_op = tf.merge_all_summaries()
	summarywriter = tf.train.SummaryWriter("./to_sum", sess.graph_def)

	sess.run(tf.initialize_all_variables())

	for step in range(max_iter):
		print "Iter %d" %(step)

		_, loss_val, acc_val = sess.run([train_step, loss, acc], feed_dict={X: X_, Y: Y_})
		summary_str = sess.run(summary_op)
		summarywriter.add_summary(summary_str, step)

		print "Loss: %g" %(loss_val)
		print "Acc: %g" %(acc_val)

def main(_):
	train_X, train_Y = get_training_data("data/train.csv")

	with tf.Session() as sess:
		training(sess, train_X, train_Y)

		#pred_Y = test(sess, test_X)

if __name__ == "__main__":
	tf.app.run()
