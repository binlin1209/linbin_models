# Copyright 2016 Sanghoon Yoon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convolutional Restrict Boltzmann Machine for Tensorflow"""
import tensorflow as tf
import os
from .core.model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

class CRBM(Model):
    def __init__(self, name,
                 input_size, input_depth,
                 weight_size, num_features, pool_size,
                 batch_size, learning_rate, is_continuous):
        self._input = None

        self.name = name
        self.write_image = getattr(FLAGS, '{}_image_summary'.format(self.name))
        self.model_path = '{}/{}'.format(FLAGS.model_path, self.name)

        # Training hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.width = self.height = input_size
        self.depth = input_depth
        self.visible_shape = [self.height, self.width, self.depth]
        self.input_shape = [self.batch_size] + self.visible_shape

        self.weight_size = weight_size
        self.num_features = num_features
        self.weight_shape = [self.weight_size, self.weight_size, self.depth, self.num_features]

        self.hidden_shape = [self.height, self.width, self.num_features]

        self.pool_size = pool_size
        self.pooled_shape = [self.height / self.pool_size, self.width / self.pool_size, self.num_features]

        self.is_continuous = is_continuous

        self.loss_func_val = None

    # Overrode abstract parts
    @property
    def input(self):
        return self._input if self._input else self.x

    @property
    def output(self):
        return self.pooled

    @property
    def ops(self):
        return self.gradient_ascent + [self.loss_func] if self.loss_func_val is None or self.loss_func_val > .09 \
            else None

    # Build overall graphs
    def build_graphs(self):
        with tf.name_scope(self.name) as _:
            self.__declare_variables()
            self.__derive_hidden()
            self.__pool()
            self.__gibbs_sampling()
            self.__gradient_ascent()
            self.__summary()

    def build_init_ops(self):
        if self.model_exists:
            self.init_ops = tf.initialize_local_variables()
        else:
            self.init_ops = [tf.initialize_local_variables(), tf.initialize_variables(self.variables)]
        self.saver = tf.train.Saver(self.variables)

    def init_variables(self, sess):
        if self.model_exists:
            print ('There is model. It restores')
            self.saver.restore(sess, self.model_path)
        sess.run(self.init_ops)

    def propagate_results(self, results):
        self.loss_func_val = results[-1]

    def save(self, sess):
        self.saver.save(sess, self.model_path)

    def set_input(self, input):
        self._input = input

    @property
    def model_exists(self):
        return os.path.exists(self.model_path)

    # Reused private methods
    @staticmethod
    def __weight_variable(shape, name):
        initial_values = tf.truncated_normal(shape, stddev=.1)
        return tf.Variable(initial_values, name=name)

    @staticmethod
    def __bias_variables(shape, name):
        initial_values = tf.zeros(shape)
        return tf.Variable(initial_values, name=name)

    @staticmethod
    def __conv2d(x, weights):
        return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def __depthwise_conv2d(x, weights):
        return tf.nn.depthwise_conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

    def __max_pool(self, x):
        shape = [1, self.pool_size, self.pool_size, 1]
        return tf.nn.max_pool(x, ksize=shape, strides=shape, padding='SAME')

    def __avg_pool(self, x):
        shape = [1, self.pool_size, self.pool_size, 1]
        return tf.nn.avg_pool(x, ksize=shape, strides=shape, padding='SAME')

    @staticmethod
    def __get_ith_element_4d(tensor, i, shape):
        return tf.slice(tensor, [i, 0, 0, 0], shape)

    def __get_ij_vis_4d(self, tensor, i, j):
        ith_vis = self.__get_ith_element_4d(tensor, i, [1] + self.visible_shape)
        return tf.slice(ith_vis, [0, 0, 0, j], [1, self.height, self.width, 1])

    def __get_ith_hid_4d(self, tensor, i):
        flat_shape = [self.height, self.width]
        ith_hid = self.__get_ith_element_4d(tensor, i, [1] + flat_shape + [self.num_features])
        return tf.reshape(ith_hid, flat_shape + [1, self.num_features])

    # Summary methods
    def __scalar_summary(self, name, scalar):
        tf.scalar_summary('{}/{}'.format(self.name, name), scalar)

    def __mean_sqrt_summary(self, name, tensor):
        self.__scalar_summary(name, tf.sqrt(tf.reduce_mean(tf.square(tensor)), name=name))

    def __image_summary(self, name, image, max_images):
        tf.image_summary('{}/{}'.format(self.name, name), image, max_images=max_images)

    def __multidimension_summary(self, name, image, shape, max_images):
        for idx in range(0, self.batch_size):
            self.__image_summary(
                '{}/{}'.format(name, idx),
                tf.transpose(tf.slice(image, [idx, 0, 0, 0], [1] + shape),
                             perm=[3, 1, 2, 0]),
                max_images)

    @staticmethod
    def __sample(prob, name):
        return tf.nn.relu(
            tf.sign(prob - tf.random_uniform(prob.get_shape(), maxval=1.)),
            name=name)


    # Declare variable
    def __declare_variables(self):
        # Bias variables
        self.bias = self.__bias_variables([self.num_features], 'bias')
        self.cias = self.__bias_variables([self.depth], 'cias')

        # Visible (input) units
        with tf.name_scope('visible') as _:
            self.x = self._input if self._input is not None \
                else tf.placeholder(tf.float32, shape=self.input_shape, name='x')
            self.vis_0 = tf.div(self.x, 255, 'vis_0')

        # Weight variables
        with tf.name_scope('weights') as _:
            self.weights = self.__weight_variable(self.weight_shape, 'weights_forward')
            self.weights_flipped = tf.transpose(
                tf.reverse(self.weights, [True, True, False, False]), perm=[0, 1, 3, 2], name='weight_back')

        self.variables = [self.bias, self.cias, self.weights]

    def __derive_hidden(self):
        # Hidden units
        with tf.name_scope('hidden') as _:
            self.signal = self.__conv2d(self.vis_0, self.weights) + self.bias
            self.hid_prob0 = tf.sigmoid(self.signal, name='hid_prob_0')
            self.hid_state0 = self.__sample(self.hid_prob0, 'hid_state_0')

    def __pool(self):
        # Pooling
        with tf.name_scope('pooling') as _:
            self.pooled_prob = tf.sub(1., tf.div(1., 1. + tf.mul(self.__avg_pool(tf.exp(self.signal)), self.pool_size * self.pool_size)), name='pooled_prob')
            self.pooled = self.__sample(self.pooled_prob, 'pooled')

    def __gibbs_sampling(self):
        # Gibbs sampling
        # Sample visible units
        with tf.name_scope('visible') as _:
            signal_back = self.__conv2d(self.hid_state0, self.weights_flipped) + self.cias
            if self.is_continuous:
                # Visible units are continuous
                normal_dist = tf.contrib.distributions.Normal(
                    mu=signal_back, sigma=1.)
                self.vis_1 = tf.reshape(
                    tf.div(normal_dist.sample_n(1), self.weight_size * self.weight_size),
                    self.input_shape, name='vis_1')
            else:
                # Visible units are binary
                vis1_prob = tf.sigmoid(signal_back, name='vis_1')
                self.vis_1 = self.__sample(vis1_prob, 'vis_1')

        # Sample hidden units
        with tf.name_scope('hidden') as _:
            self.hid_prob1 = tf.sigmoid(self.__conv2d(self.vis_1, self.weights) + self.bias, name='hid_prob_1')

    def __gradient_ascent(self):
        # Gradient ascent
        with tf.name_scope('gradient') as _:
            self.grad_bias = tf.mul(tf.reduce_mean(self.hid_prob0 - self.hid_prob1, [0, 1, 2]),
                                    self.learning_rate * self.batch_size, name='grad_bias')
            self.grad_cias = tf.mul(tf.reduce_mean(self.vis_0 - self.vis_1, [0, 1, 2]),
                                    self.learning_rate * self.batch_size, name='grad_cias')

            # TODO: Is there any method to calculate batch-elementwise convolution?
            temp_grad_weights = tf.zeros(self.weight_shape)
            hid_filter0 = tf.reverse(self.hid_prob0, [False, True, True, False])
            hid_filter1 = tf.reverse(self.hid_prob1, [False, True, True, False])
            for idx in range(0, self.batch_size):
                hid0_ith = self.__get_ith_hid_4d(hid_filter0, idx)
                hid1_ith = self.__get_ith_hid_4d(hid_filter1, idx)

                positive = [0] * self.depth
                negative = [0] * self.depth
                one_ch_conv_shape = [self.width, self.height, 1, self.num_features]
                for jdx in range(0, self.depth):
                    positive[jdx] = tf.reshape(self.__conv2d(self.__get_ij_vis_4d(self.vis_0, idx, jdx), hid0_ith),
                                               one_ch_conv_shape)
                    negative[jdx] = tf.reshape(self.__conv2d(self.__get_ij_vis_4d(self.vis_1, idx, jdx), hid1_ith),
                                               one_ch_conv_shape)
                positive = tf.concat(2, positive)
                negative = tf.concat(2, negative)
                temp_grad_weights = tf.add(temp_grad_weights,
                                           tf.slice(tf.sub(positive, negative), [0, 0, 0, 0], self.weight_shape))

            self.grad_weights = tf.mul(temp_grad_weights, self.learning_rate / (self.width * self.height))
        self.gradient_ascent = [self.weights.assign_add(self.grad_weights),
                                self.bias.assign_add(self.grad_bias),
                                self.cias.assign_add(self.grad_cias)]

    def __summary(self):
        # Summary
        with tf.name_scope('summary') as _:
            # Loss function
            self.loss_func = tf.sqrt(tf.reduce_mean(tf.square(self.vis_0 - self.vis_1)))
            self.__scalar_summary('loss_func', self.loss_func)

            # Gradients
            self.__mean_sqrt_summary('grad_weights', self.grad_weights)
            # self.__mean_sqrt_summary('grad_bias', self.grad_bias)
            # self.__mean_sqrt_summary('grad_cias', self.grad_cias)

            # Parameters
            self.__mean_sqrt_summary('weights', self.weights)
            self.__mean_sqrt_summary('bias', self.bias)
            self.__mean_sqrt_summary('cias', self.cias)

            # Images
            if self.write_image:
                if self.depth == 3:
                    visual_units = tf.concat(1, [self.vis_0, self.vis_1])

                    hid_concat_hor = tf.reshape(tf.transpose(
                        tf.reshape(self.hid_prob0,
                                   [self.batch_size, self.height, self.width, self.num_features / 2, 2]),
                        perm=[0, 1, 3, 2, 4]), [self.batch_size, self.height, self.width * self.num_features / 2, 2])
                    hid_concat = tf.reshape(tf.transpose(hid_concat_hor, perm=[0, 3, 1, 2]),
                                            [self.batch_size, 2 * self.height, self.width * self.num_features / 2, 1])
                    hid_concat = tf.concat(3, [hid_concat]*3)

                    self.__image_summary('perceptrons', tf.concat(2, [visual_units, hid_concat]), self.batch_size)

                    self.__image_summary('weight_images', tf.transpose(self.weights, perm=[3, 0, 1, 2]), self.num_features)
                else:
                    # self.__multidimension_summary('input_images', self.x, self.visible_shape, self.num_features)
                    self.__multidimension_summary('generated_images', self.vis_1, self.visible_shape, self.num_features)
                    # self.__image_summary('weight_images', tf.transpose(self.weights, perm=[3, 0, 1, 2]), self.num_features)

                pooled_concat_hor = tf.reshape(tf.transpose(
                    tf.reshape(self.pooled,
                               [self.batch_size, self.height / self.pool_size, self.width / self.pool_size, self.num_features / 2, 2]),
                    perm=[0, 1, 3, 2, 4]), [self.batch_size, self.height / self.pool_size, self.width / self.pool_size * self.num_features / 2, 2])
                pooled_concat = tf.reshape(tf.transpose(pooled_concat_hor, perm=[0, 3, 1, 2]),
                                        [self.batch_size, 2 * self.height / self.pool_size, self.width / self.pool_size * self.num_features / 2, 1])
                self.__image_summary('pooled_images', pooled_concat, self.batch_size)
