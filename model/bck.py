#author: vamshi
#July 11, 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf 
import tensorflow.contrib.layers as layers

import os,sys
from utils.loader import *

def lrelu(x, alpha):
	return tf.nn.relu(x) - alpha*tf.nn.relu(-x)


def max_pool(inp,pool_size,strides,padding="SAME",name="pool2d"):
	'''
		Max pooling operation 
	'''
	with tf.variable_scope(name):
		return tf.layers.max_pooling2d(inp,pool_size=pool_size,strides=strides,padding=padding,name=name)


class ConvVAE:
	def __init__(self, args, batch_size=64, z_dim=100, imsize=28, learning_rate = 0.0002, gamma=35):

		self.args = args
		self.d_reuse = False
		self.e_reuse = False
		self.disc_reuse = False
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		
		self.z_dim = z_dim
		self.gamma = gamma

		self.x_dim = imsize
		self.y_dim = imsize
		self.imsize = imsize
		self.n_points = self.x_dim * self.y_dim

		# Define placeholders for input and output
		self.x = tf.placeholder(tf.float32, [batch_size, self.y_dim, self.x_dim, self.args.num_channels])

		self.y = tf.placeholder(tf.float32, [batch_size, self.y_dim, self.x_dim, self.args.num_channels])

		# Create autoencoder network
		self._create_network()
		# Define loss function based variational upper-bound and corresponding optimizer
		self._create_loss_optimizer()

		# Initializing the tensorflow variables
		init = tf.global_variables_initializer()

		# Launch the session
		self.sess = tf.InteractiveSession()
		self.sess.run(init)
		self.saver = tf.train.Saver(tf.all_variables())

	def _create_network(self):

		self.z_mean, self.z_log_sigma_sq = self._encoder_network(self.x)

		# Draw one sample z from Gaussian distribution
		n_z = self.z_dim
		eps = tf.random_normal((self.batch_size, n_z), 0.0, 0.1, dtype=tf.float32)
		# z = mu + sigma*epsilon
		self.z = tf.add(self.z_mean, tf.multiply(tf.exp(self.z_log_sigma_sq/2.0), eps))

		# Use generator to determine mean of Bernoulli distribution of reconstructed input
		self.y_logits, self.y_pred = self._decoder_network(self.z)


	def _encoder_network(self, inputs):
		# Generate probabilistic encoder (recognition network), which
		# maps inputs onto a normal distribution in latent space.
		# The transformation is parametrized and can be learned.

		with tf.variable_scope("encoder"):
			e_1 = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="e_1", padding="same")
			e_1 = tf.layers.batch_normalization(e_1)

			e_2 = tf.layers.conv2d(inputs=e_1, filters=128, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="e_2", padding="same")
			e_2 = tf.layers.batch_normalization(e_2)

			#e_3 = tf.layers.conv2d(inputs=e_2, filters=32, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_3", padding="same")
			#e_3 = tf.layers.batch_normalization(e_3)

			#e_4 = tf.layers.conv2d(inputs=e_3, filters=32, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_4", padding="same")
			#e_4 = tf.layers.batch_normalization(e_4)

			e_2_reshape = tf.contrib.layers.flatten(e_2)
			e_mean = tf.layers.dense(inputs=e_2_reshape, units=self.z_dim, name="e_mean")
			e_logvar = tf.layers.dense(inputs=e_2_reshape, units=self.z_dim, name="e_logvar")
		return e_mean, e_logvar

	def _decoder_network(self,inputs):
		with tf.variable_scope("decoder",reuse=self.d_reuse):
			d_0 = tf.layers.dense(inputs=inputs, units=512, activation=tf.nn.relu, name="d_0")
			d_1 = tf.layers.dense(inputs=d_0, units=3136, activation=tf.nn.relu, name="d_1")
			d_1_reshape = tf.reshape(d_1, shape=[-1, 7, 7, 64])

			d_2 = tf.layers.conv2d_transpose(inputs=d_1_reshape, filters=128, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_2", padding="same")
			d_2 = tf.layers.batch_normalization(d_2)

			d_3 = tf.layers.conv2d_transpose(inputs=d_2, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_3", padding="same")
			d_3 = tf.layers.batch_normalization(d_3)

			d_out = tf.layers.conv2d_transpose(inputs=d_3, filters=1, kernel_size=3, strides=1, name="d_out", padding="same")

			self.d_reuse = True
		return d_out, tf.nn.sigmoid(d_out)

	def discriminator(self,z):
		with tf.variable_scope("discriminator", reuse=self.disc_reuse):
			disc_1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.leaky_relu, name="disc_1")
			disc_2 = tf.layers.dense(inputs=disc_1, units=2048, activation=tf.nn.leaky_relu, name="disc_2")
			disc_3 = tf.layers.dense(inputs=disc_2, units=2048, activation=tf.nn.leaky_relu, name="disc_3")
			disc_4 = tf.layers.dense(inputs=disc_3, units=1024, activation=tf.nn.leaky_relu, name="disc_4")
			disc_5 = tf.layers.dense(inputs=disc_4, units=512, activation=tf.nn.leaky_relu, name="disc_5")
			disc_6 = tf.layers.dense(inputs=disc_5, units=128, activation=tf.nn.leaky_relu, name="disc_6")

			logits = tf.layers.dense(inputs=disc_6, units=2, name="disc_logits")
			probabilities = tf.nn.softmax(logits)
			self.disc_reuse = True

		return logits, probabilities


  	def _create_loss_optimizer(self):

		orig_image = tf.contrib.layers.flatten(self.x, scope="o")
		new_image = tf.contrib.layers.flatten(self.y_logits, scope="r")

		#reconstruction loss
		self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(
		    tf.nn.sigmoid_cross_entropy_with_logits(logits=new_image,
		                                           labels=orig_image),
		                                           axis=1))

		#l2_dist = tf.square(orig_image-tf.contrib.layers.flatten(self.y_pred))
		#self.reconstr_loss = tf.reduce_sum(l2_dist,axis=1)
		#Kl loss
		self.vae_loss_kl = 0.5*tf.reduce_mean(tf.reduce_sum( -self.z_log_sigma_sq + tf.square(self.z_mean)+ tf.exp(self.z_log_sigma_sq), axis=1)-self.z_dim)

		#TC loss according to Factor VAE paper
		real_samples = self.z
		permuted_rows = []
		for i in range(real_samples.get_shape()[1]):
		    permuted_rows.append(tf.random_shuffle(real_samples[:, i]))
		permuted_samples = tf.stack(permuted_rows, axis=1)

		# define discriminator network to distinguish between real and permuted q(z)
		logits_real, probs_real = self.discriminator(real_samples)
		logits_permuted, probs_permuted = self.discriminator(permuted_samples)
		self.tc_regulariser = tf.abs(self.gamma * (logits_real[:, 0]  - logits_real[:, 1]))

       	#Total cost
		self.vae_cost = tf.add(self.reconstr_loss + self.vae_loss_kl, self.tc_regulariser) # average over batch

		self.disc_cost = -tf.add(0.5 * tf.reduce_mean(tf.log(probs_real[:, 0])), 0.5 * tf.reduce_mean(tf.log(probs_permuted[:, 1])), name="disc_loss")

		self.t_vars = tf.trainable_variables()
		self.enc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
		self.dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
		self.disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

		# Use ADAM optimizer
		self.vae_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.vae_cost, var_list=self.enc_vars+self.dec_vars)
		self.disc_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.disc_cost, var_list=self.disc_vars)


  	def partial_fit(self, X, Y):
	    """Train model based on mini-batch of input data.
	    	Return cost of mini-batch.
	    """

	    opt, vae_cost, vae_loss_reconstr, vae_loss_kl = self.sess.run((self.vae_optimizer, self.vae_cost, self.reconstr_loss, self.vae_loss_kl),
	                              feed_dict={self.x: X, self.y: Y})
	    opt, disc_cost = self.sess.run((self.disc_optimizer, self.disc_cost), feed_dict={self.x: X, self.y: Y})
	    
	    return vae_cost[0], tf.reduce_mean(vae_loss_reconstr).eval(), tf.reduce_mean(vae_loss_kl).eval(), disc_cost

  	def transform(self, X):
	    """Transform data by mapping it into the latent space."""
	    # Note: This maps to mean of distribution, we could alternatively
	    # sample from Gaussian distribution
	    return self.sess.run(self.z_mean, feed_dict={self.x: X})

  	def generate(self, z_mu=None):
	    """ Generate data by sampling from latent space.

	    If z_mu is not None, data for this point in latent space is
	    generated. Otherwise, z_mu is drawn from prior in latent
	    space.
	    """
	    if z_mu is None:
	        z_mu = np.random.normal(size=(self.batch_size, self.z_dim))
	    
	    return self.sess.run(self.y_pred, feed_dict={self.z: z_mu})

  	def reconstruct(self, X):
	    """ Use VAE to reconstruct given data. """
	    return self.sess.run(self.y_pred, feed_dict={self.x: X})

  	def save_model(self, checkpoint_path, epoch):
	    """ saves the model to a file """
	    self.saver.save(self.sess, checkpoint_path, global_step = epoch)

  	def load_model(self, checkpoint_path):

	    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
	    print("loading model: ",ckpt.model_checkpoint_path)
	    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
