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

class ConvVAE:
    def __init__(self, args,num_channels=1, gamma=10, imsize):

        self.args = args
        self.d_reuse = False
        self.e_reuse = False
        self.disc_reuse = False

        self.learning_rate = self.args.learning_rate
        self.batch_size = self.args.batch_size
        self.z_dim = self.args.z_dim
        self.gamma = gamma

        self.x_dim = imsize
        self.y_dim = imsize
        self.imsize = imsize
        self.n_points = self.x_dim * self.y_dim
        
        self.num_channels=num_channels
        # Define placeholders for input and output
        self.x = tf.placeholder(tf.float32, [None, self.y_dim, self.x_dim, 3])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim, self.x_dim, self.num_channels])
        self.eps = tf.placeholder(tf.float32, [None, self.z_dim])

        self._create_network()
        # Define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensorflow variables
        init = tf.global_variables_initializer()
        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        self.saver = tf.train.Saver(tf.all_variables())
        self.encoder_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))

    def _create_network(self):

        self.z_mean, self.z_log_sigma_sq = self._encoder_network(self.x)

        # Draw one sample z from Gaussian distribution
        n_z = self.z_dim
        #eps = tf.random_normal((self.batch_size, n_z), 0.0, 0.1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.exp(self.z_log_sigma_sq/2.0), self.eps))

        # Use generator to determine mean of Bernoulli distribution of reconstructed input
        self.y_logits, self.y_pred = self._decoder_network(self.z)


    def _encoder_network(self, inputs, is_train=True):
		# Generate probabilistic encoder (recognition network), which
		# maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        with tf.variable_scope("encoder"):
            e_1 = tf.layers.conv2d(inputs=inputs, filters=64*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="e_1", padding="same", trainable=is_train)
            e_1 = tf.layers.batch_normalization(e_1,trainable=is_train)

            e_2 = tf.layers.conv2d(inputs=e_1, filters=128*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="e_2", padding="same",trainable=is_train)
            e_2 = tf.layers.batch_normalization(e_2,trainable=is_train)

            #e_3 = tf.layers.conv2d(inputs=e_2, filters=128, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_3", padding="same")
            #e_3 = tf.layers.batch_normalization(e_3)

            e_4 = tf.layers.conv2d(inputs=e_2, filters=256*2, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_4", padding="same")
            e_4 = tf.layers.batch_normalization(e_4)

            e_5 = tf.layers.conv2d(inputs=e_4, filters=256*2, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_5", padding="same")
            e_6 = tf.layers.batch_normalization(e_5)

            e_7 = tf.layers.conv2d(inputs=e_6, filters=256*2, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_7", padding="same")
            e_8 = tf.layers.batch_normalization(e_7)

            e_8_reshape = tf.contrib.layers.flatten(e_8)
            e_mean = tf.layers.dense(inputs=e_8_reshape, units=self.z_dim, name="e_mean",trainable=is_train)
            e_logvar = tf.layers.dense(inputs=e_8_reshape, units=self.z_dim, name="e_logvar",trainable=is_train)
        return e_mean, e_logvar

    def _decoder_network(self,inputs, is_train=True):
        with tf.variable_scope("decoder",reuse=self.d_reuse):
            d_0 = tf.layers.dense(inputs=inputs, units=512*2, activation=tf.nn.relu, name="d_0",trainable=is_train)
            d_1 = tf.layers.dense(inputs=d_0, units=8*8*64, activation=tf.nn.relu, name="d_1",trainable=is_train)
            d_1_reshape = tf.reshape(d_1, shape=[-1, 8, 8, 64])
            d_2 = tf.layers.conv2d_transpose(inputs=d_1_reshape, filters=256*2, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_2", padding="same",trainable=is_train)
            d_2 = tf.layers.batch_normalization(d_2,trainable=is_train)
            d_3 = tf.layers.conv2d_transpose(inputs=d_2, filters=128*2, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_3", padding="same",trainable=is_train)
            d_3 = tf.layers.batch_normalization(d_3,trainable=is_train)
            d_4 = tf.layers.conv2d_transpose(inputs=d_3, filters=64*2, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_4", padding="same",trainable=is_train)
            d_4 = tf.layers.batch_normalization(d_4,trainable=is_train)
            d_5 = tf.layers.conv2d_transpose(inputs=d_4, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_5", padding="same",trainable=is_train)
            d_5 = tf.layers.batch_normalization(d_5,trainable=is_train)
            d_out = tf.layers.conv2d_transpose(inputs=d_5, filters=self.num_channels, kernel_size=3, strides=1, name="d_out", padding="same",trainable=is_train)
            self.d_reuse = True
        return d_out, tf.nn.sigmoid(d_out)

    def discriminator(self,z, is_train=True):
        with tf.variable_scope("discriminator", reuse=self.disc_reuse):
            disc_1 = tf.layers.dense(inputs=z, units=1024, activation=tf.nn.leaky_relu, name="disc_1",trainable=is_train)
            disc_2 = tf.layers.dense(inputs=disc_1, units=1024, activation=tf.nn.leaky_relu, name="disc_2",trainable=is_train)
            disc_3 = tf.layers.dense(inputs=disc_2, units=1024, activation=tf.nn.leaky_relu, name="disc_3",trainable=is_train)
            disc_4 = tf.layers.dense(inputs=disc_3, units=1024, activation=tf.nn.leaky_relu, name="disc_4",trainable=is_train)
            disc_5 = tf.layers.dense(inputs=disc_4, units=1024, activation=tf.nn.leaky_relu, name="disc_5",trainable=is_train)
            disc_6 = tf.layers.dense(inputs=disc_5, units=1024, activation=tf.nn.leaky_relu, name="disc_6",trainable=is_train)

            logits = tf.layers.dense(inputs=disc_6, units=2, name="disc_logits",trainable=is_train)
            probabilities = tf.nn.softmax(logits)
            self.disc_reuse = True

        return logits, probabilities


    def _create_loss_optimizer(self,reconstr_loss="l1"):

        orig_image = tf.contrib.layers.flatten(self.y, scope="o")
        new_image = tf.contrib.layers.flatten(self.y_logits, scope="r")

        if (reconstr_loss=="cross_entropy"):
            self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=new_image,labels=orig_image),axis=1))
        elif(reconstr_loss=="l1"):
            self.reconstr_loss= tf.reduce_mean(tf.reduce_sum(tf.abs(orig_image-new_image),axis=1)) 
        elif(reconstr_loss=="l2"):                                           
            l2_dist = tf.square(orig_image-tf.contrib.layers.flatten(self.y_pred))
            self.reconstr_loss = tf.reduce_sum(l2_dist,axis=1)

        #Kl loss
        self.vae_loss_kl = 0.5*tf.reduce_mean(tf.reduce_sum( -self.z_log_sigma_sq + tf.square(self.z_mean)+ tf.exp(self.z_log_sigma_sq), axis=1)-self.z_dim)

        #TC loss according to Factor VAE paper
        real_samples = self.z
        permuted_rows = []
        for i in range(real_samples.get_shape()[1]):
            permuted_rows.append(tf.random_shuffle(real_samples[:, i]))
        permuted_samples = tf.stack(permuted_rows, axis=1)

		# discriminator network to distinguish between real and permuted q(z)
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

        # ADAM optimizer
        self.vae_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.vae_cost, var_list=self.enc_vars+self.dec_vars)
        self.disc_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.disc_cost, var_list=self.disc_vars)


    def partial_fit(self, X, Y):
        eps =  np.random.normal(size=(self.batch_size, self.z_dim))
        opt, vae_cost, vae_loss_reconstr, vae_loss_kl = self.sess.run((self.vae_optimizer, self.vae_cost, self.reconstr_loss, self.vae_loss_kl),
                feed_dict={self.x: X, self.y: Y, self.eps:eps})
        opt, disc_cost = self.sess.run((self.disc_optimizer, self.disc_cost), feed_dict={self.x: X, self.y: Y, self.eps:eps})

        return vae_cost[0], tf.reduce_mean(vae_loss_reconstr).eval(), tf.reduce_mean(vae_loss_kl).eval(), disc_cost

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=(self.batch_size, self.z_dim))
	    
        return self.sess.run(self.y_pred, feed_dict={self.z: z_mu})

    def get_distribution(self,X):
        mean, var = self.sess.run([self.z_mean, tf.exp(self.z_log_sigma_sq)], feed_dict={self.x: X})
        return mean, var

    def reconstruct(self, X):	
        eps =  np.random.normal(size=(X.shape[0], self.z_dim))
        return self.sess.run(self.y_pred, feed_dict={self.x: X, self.eps:eps})

    def save_model(self, checkpoint_path, epoch):
        self.saver.save(self.sess, checkpoint_path, global_step = epoch)

    def load_model(self, checkpoint_path, transfer="False"):

        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print("loading model: ",ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def test_loss(self,X,Y):
        eps =  np.random.normal(size=(X.shape[0], self.z_dim))
        vae_cost, vae_loss_reconstr =  self.sess.run([self.reconstr_loss,self.vae_cost],feed_dict={self.x:X,self.y:Y,self.eps:eps})
        return vae_cost, vae_loss_reconstr

	