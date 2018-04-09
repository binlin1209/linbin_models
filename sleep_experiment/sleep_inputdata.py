# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for change the format of data, then put into the tensorflow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
import scipy.io as scio
from sklearn.model_selection import train_test_split
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed



def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(f, one_hot=False, num_classes=5):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def change_data_format(data, labels, IMAGE_SIZE=None, NUM_CHANNELS=None,
                       one_hot=False, num_classes=None,
                       PIXEL_DEPTH=None):
#  import pdb; pdb.set_trace()
  d1, d2 = data.shape
  data_1 = data.reshape(d1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
  labels = labels  ####原来的数据格式是（1，35630）取一维数据得到（35630，）
  data_1 = numpy.frombuffer(data_1).astype(numpy.float32)
  if PIXEL_DEPTH != None:
    data_1 = (data_1 - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data_1 = data_1.reshape(d1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
  labels_1 = numpy.frombuffer(labels, dtype=numpy.float64).astype(numpy.uint8)
  if one_hot:  ###把数据转成one-hot类型
    labels_1 = dense_to_one_hot(labels_1, num_classes)
  return data_1, labels_1


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert (images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape)))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   seed=None):

  subject1 = scio.loadmat('E:\\sleep_dir\\data_mat_final_new2\\mat_final.mat')
  zong_EEGPzOz = subject1["sleep_EEGPzOz"]
  zong_anno = subject1["sleep_anno"]
  zong_700EEGPzOz = subject1["sleep_700EEGPzOz"]
  zong_700anno = subject1["sleep_700anno"]

  total_EEG_1channel = zong_EEGPzOz
  total_700EEG_1channel = zong_700EEGPzOz[:7000]
  total_anno_1channel = zong_anno[0]
  total_700anno_1channel = zong_700anno[0][:7000]
  EEG_1channel = numpy.row_stack((total_EEG_1channel, total_700EEG_1channel))
  label_1channel = numpy.concatenate((total_anno_1channel, total_700anno_1channel), axis=0)
  data_train, data_test, label_train, label_test = train_test_split(EEG_1channel, label_1channel, test_size=0.7,
                                                                    random_state=0)
  data_valid, data_test, label_valid, label_test = train_test_split(data_test, label_test, test_size=0.5,
                                                                    random_state=0)

  PIXEL_DEPTH = EEG_1channel.max()   #####是否进行归一化

  train_images, train_labels = change_data_format(data_train, label_train, IMAGE_SIZE=48,
                                                NUM_CHANNELS=1, one_hot=True, num_classes=5,
                                                PIXEL_DEPTH=PIXEL_DEPTH)
  test_images, test_labels = change_data_format(data_test, label_test, IMAGE_SIZE=48,
                                                NUM_CHANNELS=1, one_hot=True, num_classes=5,
                                                PIXEL_DEPTH=PIXEL_DEPTH)
  validation_images, validation_labels = change_data_format(data_valid, label_valid, IMAGE_SIZE=48,
                                              NUM_CHANNELS=1, one_hot=True, num_classes=5,
                                              PIXEL_DEPTH=PIXEL_DEPTH)

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  
  return base.Datasets(train=train, validation=validation, test=test)




# def extract_labels(f, one_hot=False, num_classes=10):
#   """Extract the labels into a 1D uint8 numpy array [index].
#
#   Args:
#     f: A file object that can be passed into a gzip reader.
#     one_hot: Does one hot encoding for the result.
#     num_classes: Number of classes for the one hot encoding.
#
#   Returns:
#     labels: a 1D uint8 numpy array.
#
#   Raises:
#     ValueError: If the bystream doesn't start with 2049.
#   """
#   print('Extracting', f.name)
#   with gzip.GzipFile(fileobj=f) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2049:
#       raise ValueError('Invalid magic number %d in MNIST label file: %s' %
#                        (magic, f.name))
#     num_items = _read32(bytestream)
#     buf = bytestream.read(num_items)
#     labels = numpy.frombuffer(buf, dtype=numpy.uint8)
#     # labels.shape
#     # (60000,)
#     # labels[:5]
#     # array([5, 0, 4, 1, 9], dtype=uint8)
#     if one_hot:
#       return dense_to_one_hot(labels, num_classes)
#     return labels


