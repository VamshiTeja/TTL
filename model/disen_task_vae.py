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
    def __init__(self, args, A, num_channels_x, num_channels_y,imsize, gamma=10):

        self.args = args
        self.d_reuse = False
        self.e_reuse = False
        self.disc_reuse = False
        self.image_disc_reuse=False
        self.learning_rate = self.args.learning_rate
        self.batch_size = self.args.batch_size
        self.z_dim = self.args.z_dim
        self.z_selected_dim = A.shape[1]
        self.gamma = gamma

        self.x_dim = imsize
        self.y_dim = imsize
        self.imsize = imsize
        self.n_points = self.x_dim * self.y_dim
        self.num_channels_x=num_channels_x
        self.num_channels_y=num_channels_y

        # Define placeholders for input and output
        self.x = tf.placeholder(tf.float32, [None, self.y_dim, self.x_dim, self.num_channels_x])
        self.y = tf.placeholder(tf.float32, [None, self.y_dim, self.x_dim, self.num_channels_y])

        self.eps = tf.placeholder(tf.float32, [None, self.z_selected_dim])

        self.latent_selector = A
        # Create autoencoder network
        self._create_network()

        self.mi()
        # Define loss function based variational upper-bound and corresponding optimizer
        
        self._create_loss_optimizer()
        
        # Initializing the tensorflow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        self.saver = tf.train.Saver(tf.all_variables())
        self.encoder_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))
        self.mi_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='t_theta'))


    def _create_network(self):

        self.z_mean, self.z_log_sigma_sq = self._encoder_network(self.x)

        # Draw one sample z from Gaussian distribution
        n_z = self.latent_selector.shape[1]
        
        #eps = tf.random_normal((self.batch_size, n_z), 0.0, 0.1, dtype=tf.float32)
        # z = mu + sigma*epsilon

        #select latents using selector matrix(A)
        self.z_mean_transfer = tf.transpose(tf.matmul(self.latent_selector,self.z_mean, transpose_b=True, adjoint_a=True))
        self.z_log_sigma_sq_transfer = tf.transpose(tf.matmul(self.latent_selector, self.z_log_sigma_sq, transpose_b=True, adjoint_a=True))

        #self.z = tf.add(self.z_mean, tf.multiply(tf.exp(self.z_log_sigma_sq/2.0), eps))

        self.z_selected = tf.add(self.z_mean_transfer, tf.multiply(tf.exp(self.z_log_sigma_sq_transfer/2.0), self.eps))

        # Use generator to determine mean of Bernoulli distribution of reconstructed input
        self.y_logits, self.y_pred = self._decoder_network(self.z_selected)

        #Give reconstructed and original images to image discriminator

        # self.real_disc_logits, self.real_disc_prob =self.image_disc(self.y,self.y)
        # self.fake_disc_logits, self.fake_disc_prob =self.image_disc(self.y,self.y_pred)


    def _encoder_network(self, inputs, is_train=True):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.

        with tf.variable_scope("encoder"):
            e_1 = tf.layers.conv2d(inputs=inputs, filters=64*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="e_1", padding="same", trainable=is_train)
            # e_1 = tf.layers.batch_normalization(e_1,trainable=is_train)

            e_2 = tf.layers.conv2d(inputs=e_1, filters=128*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="e_2", padding="same",trainable=is_train)
            # e_2 = tf.layers.batch_normalization(e_2,trainable=is_train)

            #e_3 = tf.layers.conv2d(inputs=e_2, filters=128, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_3", padding="same")
            #e_3 = tf.layers.batch_normalization(e_3)

            e_4 = tf.layers.conv2d(inputs=e_2, filters=128*2, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_4", padding="same")
            # e_4 = tf.layers.batch_normalization(e_4)

            e_5 = tf.layers.conv2d(inputs=e_4, filters=256*2, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_5", padding="same")
            # e_6 = tf.layers.batch_normalization(e_5)

            e_7 = tf.layers.conv2d(inputs=e_5, filters=256*2, kernel_size=3, strides=1, activation=tf.nn.leaky_relu, name="e_7", padding="same")
            # e_8 = tf.layers.batch_normalization(e_7)

            e_8_reshape = tf.contrib.layers.flatten(e_7)
            e_mean = tf.layers.dense(inputs=e_8_reshape, units=self.z_dim, name="e_mean",trainable=is_train)
            e_logvar = tf.layers.dense(inputs=e_8_reshape, units=self.z_dim, name="e_logvar",trainable=is_train)
        return e_mean, e_logvar

    def _decoder_network(self,inputs, is_train=True):
        with tf.variable_scope("decoder",reuse=self.d_reuse):
            d_0 = tf.layers.dense(inputs=inputs, units=512*2, activation=tf.nn.relu, name="d_0",trainable=is_train)
            d_1 = tf.layers.dense(inputs=d_0, units=8*8*64, activation=tf.nn.relu, name="d_1",trainable=is_train)
            d_1_reshape = tf.reshape(d_1, shape=[-1, 8, 8, 64])
            d_2 = tf.layers.conv2d_transpose(inputs=d_1_reshape, filters=256*2, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_2", padding="same",trainable=is_train)
            # d_2 = tf.layers.batch_normalization(d_2,trainable=is_train)
            d_3 = tf.layers.conv2d_transpose(inputs=d_2, filters=128*2, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_3", padding="same",trainable=is_train)
            # d_3 = tf.layers.batch_normalization(d_3,trainable=is_train)
            d_4 = tf.layers.conv2d_transpose(inputs=d_3, filters=64*2, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_4", padding="same",trainable=is_train)
            # d_4 = tf.layers.batch_normalization(d_4,trainable=is_train)
            d_5 = tf.layers.conv2d_transpose(inputs=d_4, filters=64, kernel_size=3, strides=2, activation=tf.nn.relu, name="d_5", padding="same",trainable=is_train)
            # d_5 = tf.layers.batch_normalization(d_5,trainable=is_train)
            d_out = tf.layers.conv2d_transpose(inputs=d_5, filters=self.num_channels_y, kernel_size=3, strides=1, name="d_out", padding="same",trainable=is_train)
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

    def image_disc(self,x,y, is_train=True):
        '''
            x: real/fake; y:fake
        '''
        with tf.variable_scope("image_disc", reuse=self.image_disc_reuse):
            inputs=tf.concat([x,y],axis=x.get_shape().as_list()[-1])
            disc_1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="id_1", padding="same", trainable=is_train)
            # disc_1 = tf.layers.batch_normalization(disc_1,trainable=is_train)
            disc_2 = tf.layers.conv2d(inputs=disc_1, filters=64, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="id_2", padding="same", trainable=is_train)
            disc_2 = tf.layers.batch_normalization(disc_2,trainable=is_train)
            disc_3 = tf.layers.conv2d(inputs=disc_2, filters=64*2, kernel_size=3, strides=2, activation=tf.nn.leaky_relu, name="id_3", padding="same", trainable=is_train)
            disc_3 = tf.layers.batch_normalization(disc_3,trainable=is_train)
            disc_3 = tf.contrib.layers.flatten(disc_3)
            disc_4 = tf.layers.dense(inputs=disc_3, units=1024, activation=tf.nn.leaky_relu, name="idisc_4",trainable=is_train)
            disc_4 = tf.layers.batch_normalization(disc_4,trainable=is_train)
            logits = tf.layers.dense(inputs=disc_4, units=1, name="idisc_logits",trainable=is_train)
            probabilities = tf.nn.sigmoid(logits)
            self.image_disc_reuse = True
            # indices_real = 
            # d_real = logits
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
        self.vae_loss_kl = 0.5*tf.reduce_mean(tf.reduce_sum( -self.z_log_sigma_sq_transfer + tf.square(self.z_mean_transfer)+ tf.exp(self.z_log_sigma_sq_transfer), axis=1)-self.z_selected_dim)
        # self.d_loss=0.5*(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_disc_logits,labels=tf.ones_like(self.real_disc_logits)))+tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_disc_logits,labels=tf.zeros_like(self.fake_disc_logits))))
        # self.g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_disc_logits,labels=tf.ones_like(self.fake_disc_logits)))
        #TC loss according to Factor VAE paper
        real_samples = self.z_selected
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
        # self.image_disc_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='image_disc')

        self.mi_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='t_theta')

        # ADAM optimizer
        #self.image_disc_optmizer=tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=self.image_disc_vars)
        self.vae_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.vae_cost, var_list=self.enc_vars+self.dec_vars)
        self.disc_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.disc_cost, var_list=self.disc_vars)
        self.mi_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.v_lb,var_list=self.mi_vars)
        
    def partial_fit(self, X, Y):
        """Train model based on mini-batch of input data.
            Return cost of mini-batch.
        """
        eps =  np.random.normal(size=(X.shape[0], self.z_selected_dim))
        opt, vae_cost, vae_loss_reconstr, vae_loss_kl = self.sess.run((self.vae_optimizer, self.vae_cost, self.reconstr_loss, self.vae_loss_kl),
                                  feed_dict={self.x: X, self.y: Y, self.eps:eps})
        opt, disc_cost = self.sess.run((self.disc_optimizer, self.disc_cost), feed_dict={self.x: X, self.y: Y,self.eps:eps})
        
        return vae_cost[0], tf.reduce_mean(vae_loss_reconstr).eval(), tf.reduce_mean(vae_loss_kl).eval(), disc_cost


    def partial_fit_with_disc(self, X, Y):
        """Train model based on mini-batch of input data.
            Return cost of mini-batch.
        """
        eps =  np.random.normal(size=(X.shape[0], self.z_selected_dim))
        _,image_disc_cost=self.sess.run([self.image_disc_optmizer,self.d_loss],feed_dict={self.x: X, self.y: Y,self.eps:eps})
        opt, vae_cost, vae_loss_reconstr, vae_loss_kl = self.sess.run((self.vae_optimizer, self.vae_cost, self.reconstr_loss, self.vae_loss_kl),
                                  feed_dict={self.x: X, self.y: Y, self.eps:eps})
        opt, disc_cost,d_loss,g_loss = self.sess.run((self.disc_optimizer, self.disc_cost,self.d_loss,self.g_loss), feed_dict={self.x: X, self.y: Y,self.eps:eps})
        
        return vae_cost[0], tf.reduce_mean(vae_loss_reconstr).eval(), tf.reduce_mean(vae_loss_kl).eval(), disc_cost,d_loss,g_loss


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
        eps =  np.random.normal(size=(self.batch_size, self.z_selected_dim))
        return self.sess.run(self.y_pred, feed_dict={self.x: X, self.eps: eps})

    def save_model(self, checkpoint_path, epoch):
        """ saves the model to a file """
        self.saver.save(self.sess, checkpoint_path, global_step = epoch)

    def saver_mi(self, checkpoint_path, epoch):
        self.mi_saver.save(self.sess, checkpoint_path, global_step = epoch)

    def load_model(self, checkpoint_path, load_transfer=False,load_mi=False):

        if(load_mi):
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            print("loading MI Network: ",ckpt.model_checkpoint_path)
            self.mi_saver.restore(self.sess, ckpt.model_checkpoint_path)

        elif(load_transfer==False):
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            print("loading encoder from source task: ",ckpt.model_checkpoint_path)
            self.encoder_saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            print("Loading weights from: ",ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)   


    def test_loss(self,X,Y):
        eps =  np.random.normal(size=(X.shape[0], self.z_selected_dim))
        vae_cost, vae_loss_reconstr, vae_loss_kl =  self.sess.run([self.reconstr_loss,self.vae_cost, self.vae_loss_kl],feed_dict={self.x:X,self.y:Y,self.eps:eps})
        return tf.reduce_mean(vae_cost).eval(), tf.reduce_mean(vae_loss_reconstr).eval(), tf.reduce_mean(vae_loss_kl).eval()

    def theta(self,x,z,reuse=False):
        with tf.variable_scope("t_theta",reuse=reuse):
            x_ = tf.contrib.layers.flatten(x)
            z_mi=tf.layers.dense(inputs=z, units=1024,name="z_mi1",trainable=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
            z_mi=tf.layers.dense(inputs=z_mi, units=512,name="z_mi2",trainable=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

            x_mi=tf.layers.dense(inputs=x_, units=1024,name="x_mi1",trainable=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
            x_mi=tf.layers.dense(inputs=x_mi, units=512,name="x_mi2",trainable=True,kernel_initializer=tf.contrib.layers.xavier_initializer())

            t=tf.layers.dense(inputs=z_mi+x_mi, units=1,name="t_mi",trainable=True,kernel_initializer=tf.contrib.layers.xavier_initializer())
            return t

    def mi(self):
        self.t_theta = self.theta(self.y,self.z_selected)
        self.permuted_rows = []
        for i in range(self.z_selected_dim):
            self.permuted_rows.append(tf.random_shuffle(self.z_selected[:, i]))
        self.per_z =tf.stack(self.permuted_rows, axis=1)
        self.t_theta_bar=self.theta(self.y,self.per_z,reuse=True)
        self.exp_t_theta=tf.exp(self.t_theta_bar)
        self.v_lb=tf.log(tf.reduce_mean(self.exp_t_theta))-tf.reduce_mean(self.t_theta)
        
    def train_mi(self,x,y):
        self.sess.run(self.mi_opt,{self.x:x, self.y:y,self.eps:np.random.normal(size=[self.batch_size,self.z_selected_dim])})

    def mi_value(self,x,y):
        MI = self.sess.run(self.v_lb,{self.x:x, self.y:y,self.eps:np.random.normal(size=[self.batch_size,self.z_selected_dim])})
        return -1*MI  
