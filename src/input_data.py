import tensorflow.keras as keras
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np

#Uncomment to download MNIST/CIFAR/... if you have a bug
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

_Datasets = collections.namedtuple('_Datasets', ['train', 'validation', 'test'])

class _DataSet(object):

  def __init__(self,
               images,
               labels,
               dtype,
               reshape,
               num_features,
               seed):
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

    seed1, seed2 = random_seed.get_seed(seed)
    np.random.seed(seed1 if seed is None else seed2)
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      labels = labels.reshape(labels.shape[0])    
      images = images.reshape(images.shape[0], num_features)

    if dtype == dtypes.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)

    self._num_examples = images.shape[0]
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
    start = self._index_in_epoch

    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self._images[perm0]
      self._labels = self._labels[perm0]

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
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch

      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part),
                               axis=0), np.concatenate(
                                   (labels_rest_part, labels_new_part), axis=0)

    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def load_data_set(training_size, validation_size, data_set, seed=None, reshape=True, dtype=dtypes.float32):
  if data_set == "cifar10":
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    num_features = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
  if data_set == "fashion_mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    num_features = X_train.shape[1]*X_train.shape[2]
  if data_set == "mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    if not reshape:
        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
    num_features = X_train.shape[1]*X_train.shape[2]
  if "uci" in data_set.lower():
    uci_num = int(data_set[3:])
    full_data = np.load("../UCI/data" + str(uci_num) + ".pickle", allow_pickle=True)
    X_train, X_test, y_train, y_test = full_data['x_train'], full_data['x_test'], full_data['y_train'], full_data[
      'y_test']
    print("Training data shape:", X_train.shape)
    print("Training data Std:", np.std(X_train, axis=0))
    print("Training data Mean:", np.mean(X_train, axis=0))
    num_features = X_train.shape[1]

  #Permute data
  np.random.seed(seed)
  perm0 = np.arange(X_train.shape[0])
  np.random.shuffle(perm0)
  X = X_train[perm0]
  Y = y_train[perm0]

  n = int(X_train.shape[0]*training_size)
  m = int(n*validation_size)
  X_val = X[:m]
  y_val = Y[:m]
  X_train = X[m:n]
  y_train = Y[m:n]

  if "uci" in data_set.lower():
    m = np.mean(X_train, axis = 0)
    s = np.std(X_train, axis=0)
    X_train = (X_train - m)/s
    X_val = (X_val - m) / s
    X_test = (X_test - m)/s
    print("Training data shape:", X_train.shape)
    print("Training data Std:", np.std(X_train, axis=0))
    print("Training data Mean:", np.mean(X_train, axis=0))
  print("There are", X_train.shape[0], "samples in the training set.")
  print("There are", X_val.shape[0], "samples in the validation set.")

  options = dict(dtype=dtype, reshape=reshape, num_features=num_features, seed=seed)

  train = _DataSet(X_train, y_train, **options )
  validation = _DataSet(X_val, y_val, **options)
  test = _DataSet(X_test, y_test, **options)

  return _Datasets(train=train, validation=validation, test=test)