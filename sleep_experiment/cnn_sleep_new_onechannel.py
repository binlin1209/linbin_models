# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 08:22:22 2018

@author: linbin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import scipy.io as sio
import numpy as np
import os
import sys
import timeit

import numpy

import random
import numpy as np
from sklearn.model_selection import train_test_split


#####change "SAME" to "VALID"

Fs = 100

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 48    #####修改
NUM_CHANNELS = 1
#PIXEL_DEPTH = 255
NUM_LABELS = 5
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 500
NUM_EPOCHS = 5000
EVAL_BATCH_SIZE = 500
EVAL_FREQUENCY = 1000  # Number of steps between evaluations.


FLAGS = None


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32

# def maybe_download(filename)：
#     pass
#   """Download the data from Yann's website, unless it's already here."""
#   if not tf.gfile.Exists(WORK_DIRECTORY):
#     tf.gfile.MakeDirs(WORK_DIRECTORY)
#   filepath = os.path.join(WORK_DIRECTORY, filename)
#   if not tf.gfile.Exists(filepath):
#     filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
#     with tf.gfile.GFile(filepath) as f:
#       size = f.size()
#     print('Successfully downloaded', filename, size, 'bytes.')
#   return filepath


#def extract_data(filename, num_images):
#  """Extract the images into a 4D tensor [image index, y, x, channels].
#
#  Values are rescaled from [0, 255] down to [-0.5, 0.5].
#  """
#  print('Extracting', filename)
#  with gzip.open(filename) as bytestream:
#    bytestream.read(16)
#    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
#    data = numpy.frombuffer(buf).astype(numpy.float32)
##    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
#    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
#    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])


def main(_):
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
#    path_file_s = "E:\\sleep_dir\\data_mat_final_new2\\mnist_pic\\"
#    train_data_filename = path_file_s + "train-images-idx3-ubyte.gz"
#    train_labels_filename = path_file_s + "train-labels-idx1-ubyte.gz"
#    test_data_filename = path_file_s + "t10k-images-idx3-ubyte.gz"
#    test_labels_filename = path_file_s + "t10k-labels-idx1-ubyte.gz"
#    
##    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
##    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
##    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
##    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
#
#    # Extract it into numpy arrays.
#    train_data = extract_data(train_data_filename, 60000)
#    train_labels = extract_labels(train_labels_filename, 60000)
#    test_data = extract_data(test_data_filename, 10000)
#    test_labels = extract_labels(test_labels_filename, 10000)
#
#    # Generate a validation set.
#    validation_data = train_data[:VALIDATION_SIZE, ...]
#    validation_labels = train_labels[:VALIDATION_SIZE]
#    train_data = train_data[VALIDATION_SIZE:, ...]
#    train_labels = train_labels[VALIDATION_SIZE:]
    #####E:\sleep_dir\data_mat_final_new2
    subject1 = sio.loadmat('/media/re/7A308F7E308F405D/linbin/tensorflowlinbin/sleep_dir/data_mat_final_new2/mat_final.mat')
    zong_anno = subject1["sleep_anno"]
#    zong_EEGFpzCz = subject1["sleep_EEGFpzCz"]
    zong_EEGPzOz = subject1["sleep_EEGPzOz"]
#    zong_700EEGFpzCz = subject1["sleep_700EEGFpzCz"]
    zong_700EEGPzOz = subject1["sleep_700EEGPzOz"]
    zong_700anno = subject1["sleep_700anno"]
    
#    zong_EEG_2channel = np.row_stack((zong_EEGFpzCz,zong_EEGPzOz))
#    zong_anno_2channel = np.concatenate((zong_anno[0],zong_anno[0]), axis=0)
#    zong_700EEG_2channel = np.row_stack((zong_700EEGFpzCz[:7000],zong_700EEGPzOz[:7000]))
#    zong_700anno_2channel = np.concatenate((zong_700anno[0][:7000],zong_700anno[0][:7000]), axis=0)
#    EEG_2channel = np.row_stack((zong_EEG_2channel,zong_700EEG_2channel))
#    label_2channel = np.concatenate((zong_anno_2channel,zong_700anno_2channel),axis=0)
    
    total_EEG_1channel = zong_EEGPzOz
    total_700EEG_1channel = zong_700EEGPzOz[:7000]
    total_anno_1channel = zong_anno[0]
    total_700anno_1channel = zong_700anno[0][:7000]
    
    EEG_1channel = np.row_stack((total_EEG_1channel,total_700EEG_1channel))
    label_1channel = np.concatenate((total_anno_1channel,total_700anno_1channel), axis=0)
    PIXEL_DEPTH = EEG_1channel.max()
    #### data
#    import pdb; pdb.set_trace()
#    data = numpy.frombuffer(EEG_2channel).astype(numpy.float32)
#    EEG_2channel = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
##    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
#    #### label
#    label_2channel = numpy.frombuffer(label_2channel, dtype=numpy.float64).astype(numpy.int64)
    
#    import pdb; pdb.set_trace()
    
    data_train, data_test, label_train, label_test =train_test_split(EEG_1channel, label_1channel,test_size=0.7,random_state=0)
    data_valid, data_test, label_valid, label_test = train_test_split(data_test, label_test, test_size=0.5,random_state=0)
    #import pdb; pdb.set_trace()
#    test_set = (Y_test, z_test)
    d1,d2 = data_test.shape
    test_data = data_test.reshape(d1,48,48,1)
    test_labels = label_test
    test_data = numpy.frombuffer(test_data).astype(numpy.float32)
    test_data = (test_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    test_labels = numpy.frombuffer(test_labels, dtype=numpy.float64).astype(numpy.int64)
    test_data = test_data.reshape(d1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    
#    valid_set = (Y_valid, z_valid)
    d1,d2 = data_valid.shape
    validation_data = data_valid.reshape(d1,48,48,1)
    validation_labels = label_valid
    validation_data = numpy.frombuffer(validation_data).astype(numpy.float32)
    validation_data = (validation_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    validation_labels = numpy.frombuffer(validation_labels, dtype=numpy.float64).astype(numpy.int64)
    validation_data = validation_data.reshape(d1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    
#    train_set = (X_train, y_train)
    d1,d2 = data_train.shape
    train_data = data_train.reshape(d1,48,48,1)
    train_labels = label_train
#    import pdb; pdb.set_trace()
    train_data = numpy.frombuffer(train_data).astype(numpy.float32)
    train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    train_labels = numpy.frombuffer(train_labels, dtype=numpy.float64).astype(numpy.int64)
    train_data = train_data.reshape(d1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        
#    import pdb; pdb.set_trace()
    num_epochs = NUM_EPOCHS
    
    
    
  train_size = train_labels.shape[0]

  # This is where training samples and labels are fed to the graph.
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.global_variables_initializer().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 20],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([20], dtype=data_type()))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 20, 50], stddev=0.1,
      seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[50], dtype=data_type()))
  fc1_weights = tf.Variable(  # fully connected, depth 800.
      tf.truncated_normal([9 * 9 * 50, 500],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[500], dtype=data_type()))
  fc2_weights = tf.Variable(tf.truncated_normal([500, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(
      0.1, shape=[NUM_LABELS], dtype=data_type()))
  
#  import pdb; pdb.set_trace()
  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding (i.e. the output feature map has
    # the same size as the input). Note that {strides} is a 4D array whose
    # shape matches the data layout: [image index, y, x, depth].
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='VALID')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

#  import pdb; pdb.set_trace()
  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_labels_node, logits=logits))

  # L2 regularization for the fully connected parameters.
#  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
#                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # Add the regularization term to the loss.
#  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=data_type())
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      1,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
#  optimizer = tf.train.MomentumOptimizer(learning_rate,
#                                         0.9).minimize(loss,
#                                                       global_step=batch)
  # Use GradientDescentOptimizer for the optimization. add by linbin 20180324
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                       global_step=batch)
  

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)
  # add by linbin 20180324
#  train_prediction = tf.nn.sigmoid(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))
  # add by linbin 20180324
#  eval_prediction = tf.nn.sigmoid(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the optimizer to update weights.
      sess.run(optimizer, feed_dict=feed_dict)
      # print some extra information once reach the evaluation frequency
      if step % EVAL_FREQUENCY == 0:
        # fetch some extra nodes' data
#        learning_rate =0.01
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)











