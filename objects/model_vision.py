import tensorflow as tf
import numpy as np
import json
import os

from model import TensorFlowModel


# TODO: refactor the code
class BetaVAE(TensorFlowModel):
    def __init__(self, z_size=64, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, is_training=True,
                 reuse=False, gpu_mode=True, beta=1.0):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_training = is_training
        self.kl_tolerance = kl_tolerance
        self.beta = beta
        # with tf.variable_scope('conv_vae', reuse=self.reuse):
        super().__init__('conv_vae', reuse, gpu_mode)


    def _build_graph(self):

        self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

        # Encoder
        h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name="enc_conv1")
        h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name="enc_conv2")
        h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name="enc_conv3")
        h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name="enc_conv4")
        h = tf.reshape(h, [-1, 2 * 2 * 256])

        # VAE
        self.mu = tf.layers.dense(h, self.z_size, name="enc_fc_mu")
        self.logvar = tf.layers.dense(h, self.z_size, name="enc_fc_log_var")
        self.sigma = tf.exp(self.logvar / 2.0)
        self.epsilon = tf.random_normal([self.batch_size, self.z_size])
        self.z = self.mu + self.sigma * self.epsilon

        # Decoder
        h = tf.layers.dense(self.z, 4 * 256, name="dec_fc")
        h = tf.reshape(h, [-1, 1, 1, 4 * 256])
        h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name="dec_deconv1")
        h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name="dec_deconv2")
        h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name="dec_deconv3")
        self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name="dec_deconv4")

        # train ops
        if self.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            eps = 1e-6  # avoid taking log of zero

            # reconstruction loss (logistic), commented out.
            '''
            self.r_loss = - tf.reduce_mean(
              self.x * tf.log(self.y + eps) + (1.0 - self.x) * tf.log(1.0 - self.y + eps),
              reduction_indices = [1,2,3]
            )
            self.r_loss = tf.reduce_mean(self.r_loss)*64.0*64.0
            '''

            # reconstruction loss
            self.r_loss = tf.reduce_sum(
                tf.square(self.x - self.y),
                reduction_indices=[1, 2, 3]
            )
            self.r_loss = tf.reduce_mean(self.r_loss)

            # augmented kl loss per dim
            self.kl_loss = - 0.5 * tf.reduce_sum(
                (1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)),
                reduction_indices=1
            )
            #self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
            self.kl_loss = tf.reduce_mean(self.kl_loss)

            # beta = 100.0
            self.loss = self.r_loss + self.beta * self.kl_loss

            # training
            self.lr = tf.Variable(self.learning_rate, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            grads = self.optimizer.compute_gradients(self.loss)  # can potentially clip gradients here.

            self.train_op = self.optimizer.apply_gradients(
                grads, global_step=self.global_step, name='train_step')

    def encode(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x})

    def encode_mu_logvar(self, x):
        (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
        return mu, logvar

    def decode(self, z):
        return self.sess.run(self.y, feed_dict={self.z: z})
