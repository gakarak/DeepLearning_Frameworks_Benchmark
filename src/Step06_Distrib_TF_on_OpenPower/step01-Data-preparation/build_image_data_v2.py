#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_idx_path', '/tmp/idx-train.txt',
                           'Training data idx file')
tf.app.flags.DEFINE_string('validation_idx_path', '/tmp/idx-val.txt',
                           'Validation data idx file')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 2,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 2,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 2,
                            'Number of threads to preprocess the images.')
tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')

# The labels file contains a list of valid labels are held in this file.
# For example:
# $ cat ./labels.txt
# Normal
# Tumor

# idx-train and idx-val files contains path to image and label separated by commas
# Example idx-train.txt file:
# $ cat idx-train.txt | head -n6
# /home/ar/data/data_Camelyon16_2cls500/Tumor/Tumor_002_CLS_tilec__0_12703.png,Tumor
# /home/ar/data/data_Camelyon16_2cls500/Tumor/Tumor_009_tilec__0_1862.png,Tumor
# /home/ar/data/data_Camelyon16_2cls500/Normal/Normal_001_tilec_NRM_4672.png,Normal
# /home/ar/data/data_Camelyon16_2cls500/Tumor/Tumor_026_tilec__0_4367.png,Tumor
# /home/ar/data/data_Camelyon16_2cls500/Tumor/Tumor_009_tilec__0_4547.png,Tumor
# /home/ar/data/data_Camelyon16_2cls500/Normal/Normal_005_CLS_tilec_NRM_823.png,Normal

FLAGS = tf.app.flags.FLAGS

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, text, height, width):
  """Build an Example proto for an example.
  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/text': _bytes_feature(text),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""
  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()
    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})
  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _is_png(filename):
  return '.png' in filename

def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()
  # Convert any PNG to JPEG's for consistency.
  if _is_png(filename):
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)
  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3
  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               texts, labels, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.
  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
  counter = 0
  for s in xrange(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      text = texts[i]
      image_buffer, height, width = _process_image(filename, coder)
      example = _convert_to_example(filename, image_buffer, label, text, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1
      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

def _process_image_files(name, filenames, texts, labels, num_shards):
  """Process and save list of images as TFRecord of Example protos.
  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  assert len(filenames) == len(texts)
  assert len(filenames) == len(labels)
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])
  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()
  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()
  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()
  threads = []
  for thread_index in xrange(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames,
            texts, labels, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)
  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' % (datetime.now(), len(filenames)))
  sys.stdout.flush()

def _find_image_files_from_idx(path_idx, labels_file):
    with open(path_idx,'r') as fidx, open(labels_file,'r') as flbl:
        unique_labels = [ii.strip() for ii in flbl.read().splitlines()]
        tmpLines=fidx.read().splitlines()
        tmpData=np.array([ (ss[0].strip(),ss[1].strip()) for ss in [ii.split(',') for ii in tmpLines]])
        filenames=tmpData[:,0].tolist()
        texts=tmpData[:,1].tolist()
        tmpLabelsArr = np.zeros(len(texts),np.int)
        label_index=1
        for ll in unique_labels:
            tmpLabelsArr[tmpData[:,1]==ll]=label_index
            label_index+=1
        labels=tmpLabelsArr.tolist()
        #FIXME: i think this code is not required
        shuffled_index = range(len(filenames))
        random.seed(12345)
        random.shuffle(shuffled_index)
        filenames   = [filenames[i] for i in shuffled_index]
        texts       = [texts[i] for i in shuffled_index]
        labels      = [labels[i] for i in shuffled_index]
        print('Found %d JPEG files across %d labels inside %s.' %
              (len(filenames), len(unique_labels), path_idx))
        return filenames, texts, labels

def _process_dataset_from_idx(name, path_idx, num_shards, labels_file):
    filenames, texts, labels = _find_image_files_from_idx(path_idx, labels_file)
    _process_image_files(name, filenames, texts, labels, num_shards)

def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)
  # Run it!
  _process_dataset_from_idx('validation', FLAGS.validation_idx_path,
                   FLAGS.validation_shards, FLAGS.labels_file)
  _process_dataset_from_idx('train', FLAGS.train_idx_path,
                   FLAGS.train_shards, FLAGS.labels_file)

##########################
if __name__ == '__main__':
  tf.app.run()
