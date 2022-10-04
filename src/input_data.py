import tensorflow.keras as keras
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Uncomment to download MNIST/CIFAR/... if you have a bug
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context

UCI = ['ozone-level-detection-eight',
 'wall-following-robot-navigation-4',
 'wall-following-robot-navigation-24',
 'breast-cancer-wisconsin-original',
 'optical-recognition-handwritten-digits',
 'hepatitis',
 'hill-valley',
 'breast-cancer-wisconsin-prognostic',
 'libras-movement',
 'connectionist-bench',
 'qsar-biodegradation',
 'pen-based-recognition-handwritten-digits',
 'iris',
 'statlog-project-landsat-satellite',
 'skin-segmentation',
 'spambase',
 'ozone-level-detection-one',
 'echocardiogram',
 'wine',
 'breast-cancer-wisconsin-diagnostic',
 'wall-following-robot-navigation-2',
 'dermatology',
 'blood-transfusion-service-center',
 'hill-valley-noise',
 'haberman-survival',
 'balance-scale',
 'thyroid-disease-new-thyroid',
 'spectf-heart',
 'seeds',
 'thyroid-disease-ann-thyroid',
 'yeast',
 'glass-identification',
 'hayes-roth',
 'parkinsons',
 'letter-recognition',
 'climate-model-simulation-crashes',
 'banknote-authentication',
 'planning-relax',
 'ecoli',
 'connectionist-bench-sonar',
 'magic-gamma-telescope',
 'cnae-9',
 'ionosphere',
 'image-segmentation',
 'poker-hand',
 'arrhythmia']


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
      #print(images)

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
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    num_features = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

    n = int(X_train.shape[0]*training_size)
    m = int(n*validation_size)

    if training_size != 1:
        X_train, X_left, y_train, y_left = train_test_split(X_train, y_train, test_size=(X_train.shape[0] - n), random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=m, random_state=seed)

  elif data_set == "fashion_mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    num_features = X_train.shape[1]*X_train.shape[2]

    n = int(X_train.shape[0]*training_size)
    m = int(n*validation_size)

    X_train, X_left, y_train, y_left = train_test_split(X_train, y_train, test_size=(X_train.shape[0] - n), random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=m, random_state=seed)

  elif data_set == "mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    if not reshape:
        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
    num_features = X_train.shape[1]*X_train.shape[2]

    n = int(X_train.shape[0]*training_size)
    m = int(n*validation_size)

    X_train, X_left, y_train, y_left = train_test_split(X_train, y_train, test_size=(X_train.shape[0] - n), random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=m, random_state=seed)


  else:
    uci_num = UCI[int(data_set)]
    print(uci_num)
    X = np.genfromtxt("../UCI/" + str(uci_num) + "_X.csv", delimiter=',')
    Y = np.genfromtxt("../UCI/" + str(uci_num) + "_Y.csv", delimiter=',')

    if Y.min() ==1:
        Y = Y - 1

    num_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_size, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=seed)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

  options = dict(dtype=dtype, reshape=reshape, num_features=num_features, seed=seed)

  train = _DataSet(X_train, y_train, **options )
  validation = _DataSet(X_val, y_val, **options)
  test = _DataSet(X_test, y_test, **options)

  return _Datasets(train=train, validation=validation, test=test), X_train.shape
