import tensorflow as tf
import numpy as np
from argparse import Namespace


def meta_init():
	meta = Namespace()
	meta.batch_size = 128
	meta.batch_length = 256
	meta.conv1_dim = 16
	meta.conv2_dim = 32
	meta.conv3_dim = 64
	meta.screen_width = 320
	meta.screen_height = 240
	meta.screen_channel = 3
	meta.inception1_shape = [32, 48, 64, 8, 16, 16]
	meta.inception2_shape = [32, 48, 64, 8, 16, 16]
	meta.inception3_shape = [32, 48, 64, 8, 16, 16]
	meta.fc1_dim = 512
	meta.action_dim = 4
	meta.gamma = 0.9
	meta.learning_rate = 1e-5
	meta.eps_min = 1e-5
	meta.eps_decay = 0.91
	meta.exp_size = 2048
	meta.frame_interval = 1.0 / 60.0
	meta.savefile_dir = './model'
	meta.savefile_filename = './model/model.ckpt'
	meta.ratio_p = 0.1
	meta.ratio_d = 0.9

	return meta


def xavier_variable(name, shape):
	if len(shape) == 1:
		inout = np.sum(shape) + 1
	else:
		inout = np.sum(shape)

	init_range = np.sqrt(6.0 / inout)
	initializer = tf.random_uniform_initializer(-init_range, init_range)
	return tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)


def layer_conv2d(name, image, shape, strides, padding):
	kernel = xavier_variable(name + '_kernel', shape)
	bias = xavier_variable(name + '_bias', [shape[3]])
	image = tf.nn.conv2d(image, kernel, strides, padding, name=name + '_conv')
	image = tf.nn.bias_add(image, bias, name=name + '_bias')
	return image


def layer_conv_combo(name, image, shape_conv, strides_conv, padding_conv, shape_pool, strides_pool, padding_pool):
	image = layer_conv2d(name, image, shape_conv, strides_conv, padding_conv)
	image = tf.nn.pool(image, shape_pool, 'MAX', padding_pool, strides=strides_pool)
	image = tf.nn.relu(image)
	return image


def layer_fc(name, layer, shape):
	w = xavier_variable(name + '_weight', shape)
	b = xavier_variable(name + '_bias', [shape[1]])
	layer = tf.matmul(layer, w) + b
	return layer


# GoogLeNet's inception module.
# shape is array with following order
#	[0] input
#	[1] 1x1
#	[2] 3x3 reduce
#	[3] 3x3
#	[4] 5x5 reduce
#	[5] 5x5
#	[6] pooling
# returns resulting graph and final channel size
def inception(name, image, shape):
	conv1 = layer_conv2d(name + '_1x1', image, [1, 1, shape[0], shape[1]], [1, 1, 1, 1], 'SAME')
	conv3_reduce = layer_conv2d(name + '_3x3_reduce', image, [1, 1, shape[0], shape[2]], [1, 1, 1, 1], 'SAME')
	conv3 = layer_conv2d(name + '_3x3', image, [3, 3, shape[0], shape[3]], [1, 1, 1, 1], 'SAME')
	conv5_reduce = layer_conv2d(name + '_5x5_reduce', image, [1, 1, shape[0], shape[4]], [1, 1, 1, 1], 'SAME')
	conv5 = layer_conv2d(name + '_5x5', image, [5, 5, shape[0], shape[5]], [1, 1, 1, 1], 'SAME')
	pool = tf.nn.pool(image, [3, 3], 'MAX', 'SAME', strides=[1, 1], name=name + '_pool')
	pool_reduce = layer_conv2d(name + '_pool_reduce', pool, [1, 1, shape[0], shape[6]], [1, 1, 1, 1], 'SAME')
	channel_size = shape[1] + shape[3] + shape[5] + shape[6]
	return tf.concat([conv1, conv3, conv5, pool_reduce], axis=3), channel_size


class Agent:

	def __init__(self, meta):

		self.obs_prev = tf.placeholder(tf.uint8, [None, meta.screen_height, meta.screen_width, meta.screen_channel])

		self.obs_next = tf.placeholder(tf.uint8, [None, meta.screen_height, meta.screen_width, meta.screen_channel])

		self.obs_prev_float = tf.cast(self.obs_prev, tf.float32)
		self.obs_next_float = tf.cast(self.obs_next, tf.float32)

		with tf.variable_scope('model'):
			self.Q_prev = self.build_graph(self.obs_prev_float, meta)
		with tf.variable_scope('model', reuse=True):
			self.Q_next = self.build_graph(self.obs_next_float, meta)

		self.action = tf.placeholder(tf.float32, [None, meta.action_dim])
		self.reward = tf.placeholder(tf.float32, [None, ])

		self.Q_current = tf.reduce_sum(tf.multiply(self.Q_prev, self.action), axis=1)
		self.Q_target = self.reward + meta.gamma * tf.reduce_max(self.Q_next, axis=1)

		self.decision = tf.nn.softmax(self.Q_prev)

		self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q_current))
		with tf.variable_scope('model'):
			self.global_step = tf.get_variable('global_step', trainable=False, initializer=tf.constant(0))
			self.train = tf.train.AdamOptimizer(meta.learning_rate).minimize(self.loss, self.global_step)

		self.sess = tf.Session()
		self.eps = 1

		with tf.variable_scope('model'):
			self.saver = tf.train.Saver(tf.global_variables())
			ckpt = tf.train.get_checkpoint_state(meta.savefile_dir)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				self.saver.restore(self.sess, ckpt.model_checkpoint_path)
				# Restore epsilon
				step = self.sess.run(self.global_step)
				self.eps = np.power(meta.eps_decay, step / meta.batch_length)
			else:
				self.sess.run(tf.global_variables_initializer())

		self.exp_ind = 0
		self.exp_full = False
		self.exp_obs_prev = np.empty([meta.exp_size, meta.screen_height, meta.screen_width, meta.screen_channel])
		self.exp_obs_next = np.empty([meta.exp_size, meta.screen_height, meta.screen_width, meta.screen_channel])
		self.exp_action = np.empty([meta.exp_size, meta.action_dim])
		self.exp_reward = np.empty([meta.exp_size])

		self.meta = meta


	def build_graph(self, image, meta):
		# 320 x 240
		conv1 = layer_conv_combo('conv1', image, [5, 5, 3, meta.conv1_dim], [1, 2, 2, 1], 'SAME', [2, 2], [2, 2], 'SAME')
		# 80 x 60
		conv2 = layer_conv_combo('conv2', conv1, [3, 3, meta.conv1_dim, meta.conv2_dim], [1, 1, 1, 1], 'SAME', [2, 2], [2, 2], 'SAME')
		# 40 x 30
		conv3 = layer_conv_combo('conv3', conv2, [3, 3, meta.conv2_dim, meta.conv3_dim], [1, 1, 1, 1], 'SAME', [2, 2], [2, 2], 'SAME')
		# 20 x 15
		inc1, dim = inception('inception1', conv3, [meta.conv3_dim, ] + meta.inception1_shape)
		inc2, dim = inception('inception2', inc1, [dim, ] + meta.inception2_shape)
		inc3, dim = inception('inception3', inc2, [dim, ] + meta.inception3_shape)
		pool4 = tf.nn.pool(inc3, [15, 20], 'AVG', 'VALID', strides=[15, 20], name='pool4')

		flat = tf.reshape(pool4, [-1, dim], name='flatten')
		fc1 = layer_fc('fc1', flat, [dim, meta.fc1_dim])
		relu1 = tf.nn.relu(fc1)
		fc2 = layer_fc('fc2', fc1, [meta.fc1_dim, meta.action_dim])

		return fc2


	def feed(self, obs_prev, action, reward, obs_next):
		self.exp_obs_prev[self.exp_ind] = obs_prev
		self.exp_action[self.exp_ind] = action
		self.exp_reward[self.exp_ind] = reward
		self.exp_obs_next[self.exp_ind] = obs_next
		self.exp_ind += 1

		if self.exp_ind >= self.meta.exp_size:
			self.exp_ind = 0
			self.exp_full = True
			return True
		else:
			return False


	def flush(self):
		self.exp_ind = 0
		self.exp_full = False


	def batch(self):
		if self.exp_full:
			indices = np.random.choice(self.meta.exp_size, self.meta.batch_size)
			feed = {
				self.obs_prev : self.exp_obs_prev[indices], 
				self.action : self.exp_action[indices], 
				self.reward : self.exp_reward[indices], 
				self.obs_next : self.exp_obs_next[indices]
			}
			loss_value, _ = self.sess.run([self.loss, self.train], feed_dict=feed)
			return loss_value
		else:
			print("Experience queue is not filled yet")
			return 0.0


	def decide(self, obs):
		if np.random.random() < self.eps:
			act_ind = np.random.randint(self.meta.action_dim)
		else:
			obs = np.reshape(obs, [1, ] + list(obs.shape))
			feed = {self.obs_prev : obs}
			# Use softmaxed Q as probability distribution of action
			pd = self.sess.run(self.decision, feed_dict=feed)
			pd = np.reshape(pd, [-1])
			act_cand = range(self.meta.action_dim)
			act_ind = np.random.choice(act_cand, 1, p=pd)
			act_ind = act_ind[0] # De-array-fy
		act = np.zeros(self.meta.action_dim)
		act[act_ind] = 1
		return act


	def decay(self):
		if self.eps > self.meta.eps_min:
			self.eps = self.eps * self.meta.eps_decay


	def save(self):
		self.saver.save(self.sess, self.meta.savefile_filename, global_step=self.global_step)






