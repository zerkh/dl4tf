import tensorflow as tf
import numpy as np
import argparser

def prepare_data(data_path, vocab_size, num_steps):
	word_dict = {}
	inv_word_dict = {}

	word_dict[0] = "EOF"
	word_dict[1] = "UNK"
	inv_word_dict["EOF"] = 0
	inv_word_dict["UNK"] = 1
	idx = 2
	train_X,train_Y,test_X,test_Y = [],[],[],[]

	X = []
	with open(data_path) as fin:
		lines = fin.readlines()

		for line in lines:
			X.append(line.strip().split(" "))

	for x in X:
		for w in x:
			if x not in word_dict.values():
				word_dict[idx] = x
				inv_word_dict[x] = idx

	for i in range(num_steps, len(word_dict)):
		del inv_word_dict[word_dict[i]]
		del word_dict[i]
	
	idx_X = []
	for x in X:
		idx_x = []
		for i in xrange(len(x)):
			if x[i] in inv_word_dict:
				idx_x.append(inv_word_dict[x[i]])
			else:
				idx_x.append(2)
		idx_X.append(idx_x)

	train_X = np.zeros((len(idx_X), num_steps, vocab_size))
	train_Y = np.zeros((len(idx_X), num_steps, vocab_size))
	for i in xrange(len(idx_X)):
		x = idx_X[i]
		for step in xrange(num_steps):
			if step >= len(x):
				train_X[i][step][0] = 1.0
				train_Y[i][step][0] = 1.0
				continue

			train_X[i][step][x[step]] = 1.0
			if step+1 == len(x):
				train_Y[i][step][0] = 1.0
			else:
				train_Y[i][step][x[step+1]] = 1.0

	test_X = train_X
	test_Y = train_Y

	return train_X,train_Y,test_X,test_Y,word_dict

def train(sess, train_X, train_Y, num_steps, batch_size, input_size, state_size, vocab_size, max_step=100):
	cell = LSTMCell(input_size, state_size, vocab_size)
	_input = tf.placeholder(tf.int64, [batch_size, num_steps, vocab_size])
	_target = tf.placeholder(tf.float64, [batch_size, num_steps, vocab_size])	

	outputs = []
	states = []
	_init_state = cell.zero_state(batch_size)
	state = _init_state

	for time_step in xrange(num_steps):
		state, output = cell(_input[:,time_step,:], state)
	res_out = tf.reshape(outputs, [batch_size, num_steps, vocab_size])
	loss = tf.nn.softmax_cross_entropy_with_logits(res_out, _target)
	cost = tf.reduce_sum(loss)/batch_size

	optimizer = tf.train.GradientDescentOptimizer(0.001)
	train_step = optimizer.minimize(cost)

	for step in xrange(max_step):
		batch_len = len(train_X) / batch_size
		total_cost = 0.0
		for i in xrange(batch_len):
			if i+1 != batch_len:
				X = train_X[i*batch_size:(i+1)*batch_size, :, :]
				Y = train_Y[i*batch_size:(i+1)*batch_size, :, :]
			else:
				X = train_X[i*batch_size:len(train_X), :, :]
				Y = train_Y[i*batch_size:len(train_X), :, :]

			_, cost_val = sess.run(train_step, feed_dict={_input=X, _target=Y})
			total_cost += cost_val
		print "Iter %d cost: %g" %(step, total_cost)

def test(sess, test_X, word_dict):
	cell = LSTMCell(input_size, state_size, vocab_size)
	_input = tf.placeholder(tf.int64, [batch_size, num_steps, vocab_size])

	outputs = []
	states = []
	_init_state = cell.zero_state(batch_size)
	state = _init_state

	for time_step in xrange(num_steps):
		state, output = cell(_input[:,time_step,:], state)
	res_out = tf.reshape(outputs, [batch_size, num_steps, vocab_size])

	for x in test_X:
		out = sess.run(res_out, feed_dict={_input:x})
		print out
	

def main(_):
	parser = argparser.ArgumentParser()
	parser.add_argument("--input_size", default=100, type=int)
	parser.add_argument("--vocab_size", default=30000, type=int)
	parser.add_argument("--state_size", default=100, type=int)
	parser.add_argument("--data_dir", type=str, required=True)
	parser.add_argument("--batch_size", default=50, type=int)
	parser.add_argument("--num_steps", default=35, type=int)
	option = parser.parse_args()

	num_steps = option.num_steps
	data_path = option.data_dir
	input_size = option.input_size
	output_size = option.vocab_size
	state_size = option.state_size
	batch_size = option.batch_size

	train_X, train_Y, test_X, test_Y, word_dict = prepare_data(data_path, vocab_size, num_steps)

	with tf.Session() as sess:
		train(sess, train_X, train_Y, num_steps, batch_size, input_size, state_size, vocab_size)

		test(sess, test_X, word_dict)

if __name__ == "__main__":
	tf.app.run()