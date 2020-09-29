import keras
import collections
from tensorflow.python.framework import dtypes
import numpy

_Datasets = collections.namedtuple('_Datasets', ['train', 'validation', 'test'])

class _DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype,
               reshape,
               num_subsets,
               subset_ratio,
               num_features):
    """Construct a _DataSet.

    Args:
      images: The images
      labels: The labels
      dtype: Output image dtype. One of [uint8, float32]. `uint8` output has
        range [0,255]. float32 output has range [0,1].
      reshape: Bool. If True returned images are returned flattened to vectors.
      num_subsets: Number of training subsets for stability
      subset_ratio: fraction of original training set that must be in each subset.
    """
    self.num_subsets = num_subsets
    self.subset_size = int(subset_ratio*images.shape[0])

     # Convert shape from [num examples, rows, columns, depth]
     # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      labels = labels.reshape(labels.shape[0])
      images = images.reshape(images.shape[0], num_features)

    if dtype == dtypes.float32:
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels

    self.MC_images = [0 for k in range(self.num_subsets)]
    self.MC_labels = [0 for k in range(self.num_subsets)]

    for k in range(self.num_subsets):
      perm = numpy.arange(images.shape[0])
      numpy.random.shuffle(perm)

      self.MC_images[k] = images[perm][:self.subset_size]
      self.MC_labels[k] = labels[perm][:self.subset_size]

    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

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
    subset_batch_size = int(batch_size/self.num_subsets)
    start = self._index_in_epoch

    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self.subset_size)
      numpy.random.shuffle(perm0)
      self.MC_images = [self.MC_images[k][perm0] for k in range(self.num_subsets)]
      self.MC_labels = [self.MC_labels[k][perm0]for k in range(self.num_subsets)]

    # Go to the next epoch
    if start + subset_batch_size > self.subset_size:

      # Finished epoch
      self._epochs_completed += 1

      # Get the rest examples in this epoch
      rest_num_examples = self.subset_size - start
      images_rest_part = [self.MC_images[k][start:self.subset_size] for k in range(self.num_subsets)]
      labels_rest_part = [self.MC_labels[k][start:self.subset_size] for k in range(self.num_subsets)]

      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self.subset_size)
        numpy.random.shuffle(perm)
        self.MC_images = [self.MC_images[k][perm] for k in range(self.num_subsets)]
        self.MC_labels = [self.MC_labels[k][perm] for k in range(self.num_subsets)]

      # Start next epoch
      start = 0
      self._index_in_epoch = subset_batch_size - rest_num_examples
      end = self._index_in_epoch

      batch_images = numpy.concatenate((images_rest_part[0], self.MC_images[0][start:end]), axis=0)
      batch_labels = numpy.concatenate((labels_rest_part[0], self.MC_labels[0][start:end]), axis=0)

      for k in range(1, self.num_subsets):
        images_new_part = self.MC_images[k][start:end]
        labels_new_part = self.MC_labels[k][start:end]
        batch_images = numpy.concatenate((batch_images, images_rest_part[k], images_new_part), axis=0)
        batch_labels = numpy.concatenate((batch_labels,labels_rest_part[k], labels_new_part), axis=0)

      return batch_images, batch_labels

    else:

      self._index_in_epoch += subset_batch_size
      end = self._index_in_epoch

      batch_images = self.MC_images[0][start:end]
      batch_labels = self.MC_labels[0][start:end]

      for k in range(1,self.num_subsets):
        batch_images = numpy.concatenate((batch_images, self.MC_images[k][start:end]), axis=0)
        batch_labels = numpy.concatenate((batch_labels, self.MC_labels[k][start:end]), axis=0)

      return batch_images, batch_labels

def load_data_set(num_subsets, subset_ratio, validation_size, data_set, reshape=True, dtype=dtypes.float32):
  if data_set == "cifar":
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    num_featurses = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
  if data_set == "mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    num_features = X_train.shape[1]*X_train.shape[2]
  X_val = X_train[:validation_size]
  y_val = y_train[:validation_size]
  X_train = X_train[validation_size:]
  y_train = y_train[validation_size:]

  options = dict(dtype=dtype, reshape=reshape, num_subsets=num_subsets, subset_ratio=subset_ratio, num_features=num_features)

  train = _DataSet(X_train, y_train, **options )
  validation = _DataSet(X_val, y_val, **options)
  test = _DataSet(X_test, y_test, **options)

  return _Datasets(train=train, validation=validation, test=test)
