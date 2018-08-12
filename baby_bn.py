## Original code is from mnist tutorial provided by tensorflow ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
import tensorflow as tf
epsilon = 0.001

FLAGS = None
def main(_):
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)

	x = tf.placeholder(tf.float32, [None, 784])
	# Define loss and optimizer
	# first layer
	layer1 = tf.layers.dense(x, 256, activation="sigmoid", use_bias = True)
	layer2 = tf.layers.dense(layer1, 256, activation="sigmoid", use_bias = True)
	layer3 = tf.layers.dense(layer2, 10, activation=None, use_bias = True)

	layer1_BN = tf.layers.dense(x, 256, activation=None, use_bias = True)
	mean1, var1 = tf.nn.moments(layer1_BN, [0])
	z1_hat = (layer1_BN - mean1) / tf.sqrt(var1 + epsilon)
	gamma1 = tf.Variable(tf.ones([256]))
	beta1 = tf.Variable(tf.ones([256]))
	scaled_z1 = tf.nn.sigmoid(gamma1 * z1_hat + beta1)

	layer2_BN = tf.layers.dense(scaled_z1, 256, activation=None, use_bias = True)
	mean2, var2 = tf.nn.moments(layer2_BN, [0])
	z2_hat = (layer2_BN - mean2) / tf.sqrt(var2 + epsilon)
	gamma2 = tf.Variable(tf.ones([256]))
	beta2 = tf.Variable(tf.ones([256]))
	scaled_z2 = tf.nn.sigmoid(gamma2	 * z2_hat + beta2)

	layer3_BN = tf.layers.dense(scaled_z2, 10, activation=None, use_bias = True)

	y = layer3
	y_Batch = layer3_BN

	y_ = tf.placeholder(tf.int64, [None])

	# raw formulation of cross entropy
	## nested reduce_sum sum over product of each class difference
	## for example
	## y_ = [1, 0, 0, 0], y = [0.7, 0.1, 0.1, 0.1]
	## reduce_sum = 1*log(0.7) + 0*log(0.1) + 0*log(0.1) + 0*log(0.1)
	# unstable...
	#cross_entropy = tf.reduce_mean( -tf.reduce_sum( tf.cast(y_, tf.float32) * tf.log(tf.nn.softmax(y)), reduction_indices=[1] ))
	cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
	cross_entropy_batch = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_Batch)
	#cross_entropy_batch = tf.reduce_mean( -tf.reduce_sum( tf.cast(y_, tf.float32) * tf.log(tf.nn.softmax(y_Batch)), reduction_indices=[1] ))
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
	train_step_batch = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_batch)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	correct_prediction = tf.equal(tf.argmax(y, 1), y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	correct_prediction2 = tf.equal(tf.argmax(y_Batch, 1), y_)
	accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
	acc = []
	acc_BN = []
	for i in range(10000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		#encoded = to_categorical(list(batch_ys), num_classes=10)
		_, train_loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_ : batch_ys})
		_batch, train_loss_batch = sess.run([train_step_batch, cross_entropy_batch], feed_dict={x: batch_xs, y_ : batch_ys})
		#print(train_loss)
		if i % 50 == 0:
			acc1, acc2 = sess.run([accuracy, accuracy2], feed_dict={x: mnist.test.images, y_:mnist.test.labels})
			acc.append(acc1)
			acc_BN.append(acc2)

	fig, ax = plt.subplots()
	ax.plot(range(0, len(acc)*50, 50), acc, label='WithoutBN')
	ax.plot(range(0, len(acc)*50, 50), acc_BN, label='WithBN')
	ax.set_xlabel('Training steps')
	ax.set_ylabel('Acc')
	ax.set_title('comparison')
	ax.legend(loc=4)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--data_dir',
	  type=str,
	  default='/tmp/tensorflow/mnist/input_data',
	  help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)