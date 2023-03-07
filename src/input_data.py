import tensorflow.keras as keras
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import json

# Load config file with static parameters
with open('config.json') as config_file:
        config = json.load(config_file)

  UCI =config["UCI"]
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


Last login: Tue Mar  7 08:31:27 on ttys000
(base) kimvc@FVFZT22FL416 ~ % !ssh
ssh kimvc@txe1-login.mit.edu


Last login: Mon Mar  6 15:14:22 2023 from 10.29.101.129
LLGrid (TX-E1) login Node
     ==========================================================================
    |        MIT SuperCloud High Performance Research Computing                 |
    |                                                                           |
    |  Website:	                https://supercloud.mit.edu                      |	
    |  Web Portal:	        https://txe1-portal.mit.edu                     |	
    |  Consulting:              supercloud@mit.edu	                        |
    |  Practical HPC Course:    https://learn.llx.edly.io/course/practical-hpc  |
     ==========================================================================

       **********************************************************************
      *                 === IMPORTANT POLICY INFORMATION ===                  *
      *                                                                       *
      * - Account and password sharing is prohibited.                         *
      * - Sharing home directories is prohibited, please request a shared     *
      *   group directory.                                                    *
      * - Use of MIT SuperCloud resources in violation of United States       *
      *   export control laws and regulations is prohibited.                  *
      * - Authorized users must adhere to ALL MIT Acceptable Use Policies:    *
      *               https://ist.mit.edu/about/it-policies                   *
      *                                                                       *
      *                ===  STORAGE IS NOT BACKED UP ===                      *
      *                                                                       *
      * - SuperCloud storage is not backed up on any outside storage system.  *
      * - Users should backup their critical files to other storage systems.  * 
      *   Highest priority should be given to:                                *
      *    - Software you have written that is not stored elsewhere. Consider *
      *        using GitHub or https://github.mit.edu                         *
      *    - Data you have collected that is not stored elsewhere.            *
      *    Lower priority can be given to data that can be regenerated by     *
      *    rerunning your programs.                                           *
       ***********************************************************************
kimvc@login-3:~$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       

kimvc@login-3:~$ 
kimvc@login-3:~$ 
kimvc@login-3:~$ 
kimvc@login-3:~$ 
kimvc@login-3:~$ 
kimvc@login-3:~$ 
kimvc@login-3:~$ cd HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example/
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim training_err.txt 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22084077
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22084079
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22084080
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22084081
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22084081_[0]       22084081   train_all_model kimvc   N/A                 xeon-p8    24    (null)    160G        PD  (None)                        
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22084081_[0]       22084081   train_all_model kimvc   N/A                 xeon-p8    24    (null)    160G        PD  (Priority)                    
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22084040           22084040   JupyterNotebook kimvc   2023-03-07T08:45:29 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-3:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ client_loop: send disconnect: Broken pipe
(base) kimvc@FVFZT22FL416 ~ % 
(base) kimvc@FVFZT22FL416 ~ % 
(base) kimvc@FVFZT22FL416 ~ % 
(base) kimvc@FVFZT22FL416 ~ % !ssh
ssh kimvc@txe1-login.mit.edu


Last login: Tue Mar  7 08:31:33 2023 from 66.31.42.54
LLGrid (TX-E1) login Node
     ==========================================================================
    |        MIT SuperCloud High Performance Research Computing                 |
    |                                                                           |
    |  Website:	                https://supercloud.mit.edu                      |	
    |  Web Portal:	        https://txe1-portal.mit.edu                     |	
    |  Consulting:              supercloud@mit.edu	                        |
    |  Practical HPC Course:    https://learn.llx.edly.io/course/practical-hpc  |
     ==========================================================================

       **********************************************************************
      *                 === IMPORTANT POLICY INFORMATION ===                  *
      *                                                                       *
      * - Account and password sharing is prohibited.                         *
      * - Sharing home directories is prohibited, please request a shared     *
      *   group directory.                                                    *
      * - Use of MIT SuperCloud resources in violation of United States       *
      *   export control laws and regulations is prohibited.                  *
      * - Authorized users must adhere to ALL MIT Acceptable Use Policies:    *
      *               https://ist.mit.edu/about/it-policies                   *
      *                                                                       *
      *                ===  STORAGE IS NOT BACKED UP ===                      *
      *                                                                       *
      * - SuperCloud storage is not backed up on any outside storage system.  *
      * - Users should backup their critical files to other storage systems.  * 
      *   Highest priority should be given to:                                *
      *    - Software you have written that is not stored elsewhere. Consider *
      *        using GitHub or https://github.mit.edu                         *
      *    - Data you have collected that is not stored elsewhere.            *
      *    Lower priority can be given to data that can be regenerated by     *
      *    rerunning your programs.                                           *
       ***********************************************************************
kimvc@login-4:~$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22085219           22085219   JupyterNotebook kimvc   2023-03-07T11:04:13 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-4:~$ cd HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example/
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22085374
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22085379
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22085385
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ vim train_all_models.sh
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLsub train_all_models.sh
Submitted batch job 22085387
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22085219           22085219   JupyterNotebook kimvc   2023-03-07T11:04:13 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22085387_[0-1]     22085387   train_all_model kimvc   N/A                 xeon-p8    24    (null)    160G        PD  (None)                        
22085385_[5-8]     22085385   train_all_model kimvc   N/A                 xeon-p8    24    (null)    160G        PD  (None)                        
22085379_[5-8]     22085379   train_all_model kimvc   N/A                 xeon-p8    24    (null)    160G        PD  (None)                        
22085374_[0-1]     22085374   train_all_model kimvc   N/A                 xeon-p8    24    (null)    160G        PD  (None)                        
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22085219           22085219   JupyterNotebook kimvc   2023-03-07T11:04:13 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22085374_[0-1]     22085374   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22085379_[5-8]     22085379   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22085385_[5-8]     22085385   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22085387_[0-1]     22085387   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ 
kimvc@login-4:~/HAIM_H2O_shared/TabText/Examples/Harford_LOS_Example$ cd 
kimvc@login-4:~$ cd final_HDL/HDL/src/
kimvc@login-4:~/final_HDL/HDL/src$ vim find_tree.py
kimvc@login-4:~/final_HDL/HDL/src$ vim find_tree.py
kimvc@login-4:~/final_HDL/HDL/src$ vim find_tree.sh
kimvc@login-4:~/final_HDL/HDL/src$ LLsub find_tree.sh
Submitted batch job 22085682
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ LLstat
LLGrid: txe1 (running slurm-wlm 20.11.9)
JOBID              ARRAY_JOB_ NAME            USER    START_TIME          PARTITION  CPUS  FEATURES  MIN_MEMORY  ST  NODELIST(REASON)              
22085219           22085219   JupyterNotebook kimvc   2023-03-07T11:04:13 jupyter-cp 2     (null)    4000M       R   d-5-1-1                       
22085682_[0-10]    22085682   find_tree.sh    kimvc   N/A                 xeon-g6-vo 1     (null)    64G         PD  (None)                        
22085374_[0-1]     22085374   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22085379_[5-8]     22085379   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22085385_[5-8]     22085385   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22085387_[0-1]     22085387   train_all_model kimvc   2024-03-04T02:17:24 xeon-p8    24    (null)    160G        PD  (Priority)                    
22056570_2         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-5-5-3                       
22056570_3         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056570_4         22056570   train_all_model kimvc   2023-03-05T02:17:24 xeon-p8    24    (null)    64G         R   d-6-4-2                       
22056561_2         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_3         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-16-14-4                     
22056561_4         22056561   train_all_model kimvc   2023-03-05T02:16:54 xeon-p8    24    (null)    64G         R   c-17-3-2                      
22083039_0         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-3-8-2                       
22083039_1         22083039   train_all_model kimvc   2023-03-07T02:10:11 xeon-p8    24    (null)    100G        R   d-4-8-3                       
22083038_8         22083038   train_all_model kimvc   2023-03-07T02:06:47 xeon-p8    24    (null)    100G        R   d-5-12-1                      
22083038_7         22083038   train_all_model kimvc   2023-03-07T02:05:44 xeon-p8    24    (null)    100G        R   c-17-13-3                     
22083038_6         22083038   train_all_model kimvc   2023-03-07T01:53:21 xeon-p8    24    (null)    100G        R   d-5-7-4                       
22083038_5         22083038   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   d-19-10-4                     
22083037_8         22083037   train_all_model kimvc   2023-03-07T01:52:49 xeon-p8    24    (null)    100G        R   c-16-3-2                      
22083033_5         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-9-3                      
22083034_0         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-5-3                       
22083035_0         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-6-4-3                       
22083037_5         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-10-2                     
22083033_6         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-16-14-2                     
22083033_7         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   c-17-8-1                      
22083033_8         22083033   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-3-9-3                       
22083034_1         22083034   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-4-11-1                      
22083035_1         22083035   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-4-1                      
22083037_6         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-16-11-2                     
22083037_7         22083037   train_all_model kimvc   2023-03-07T01:52:17 xeon-p8    24    (null)    100G        R   d-17-2-1                      
22083031_1         22083031   train_all_model kimvc   2023-03-07T01:50:34 xeon-p8    24    (null)    100G        R   d-5-5-3                       
22084081_0         22084081   train_all_model kimvc   2023-03-07T08:52:03 xeon-p8    24    (null)    160G        R   d-3-13-4                      
22084079_0         22084079   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-2-2                       
22084080_0         22084080   train_all_model kimvc   2023-03-07T08:51:33 xeon-p8    24    (null)    160G        R   d-3-7-2                       
22084077_0         22084077   train_all_model kimvc   2023-03-07T08:51:02 xeon-p8    24    (null)    160G        R   c-16-9-4                      

kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ 
kimvc@login-4:~/final_HDL/HDL/src$ ls
0.csv     UCI_time.csv  error.txt     found_best_tree.csv  input_data.py         output.text   pats_out.txt                        results_MLP_model0.csv   results_MLP_model36.csv  saved_weights.pkl  trees.ipynb
LICENSE   __pycache__   find_tree.py  found_good_tree.csv  kim.csv               outputs       pgd_attack.py                       results_MLP_model12.csv  results_MLP_model39.csv  train.py           uci
Networks  config.json   find_tree.sh  found_tree.csv       l0_regularization.py  pats_err.txt  results_CNN_modelfashion_mnist.csv  results_MLP_model27.csv  run_cifar_kim.sh         train_time.py      uci_aggregated_Leo_fixed_again.csv
kimvc@login-4:~/final_HDL/HDL/src$ LLsub -i -g volta:1
 
 
NOTE: For interactive sessions, the default time limit is set to 24 hours.
      We recommend to run your session as a batch job if it takes longer than the default limit.
 
salloc -p xeon-g6-volta --gres=gpu:volta:1 --constraint=xeon-g6 --time=24:00:00 --qos=high  srun  --pty bash -i
salloc: Pending job allocation 22086385
salloc: job 22086385 queued and waiting for resources
salloc: job 22086385 has been allocated resources
salloc: Granted job allocation 22086385
salloc: Waiting for resource configuration
salloc: Nodes d-14-5-2 are ready for job
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim run_cifar_kim.sh 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train_time.py --train_size 1 --data_set cifar10  --network_type ALEX --exp_id -1
Traceback (most recent call last):
  File "train_time.py", line 12, in <module>
    import importlib.util
ImportError: No module named util
kimvc@d-14-5-2:~/final_HDL/HDL/src$ module load anaconda/2022a
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train_time.py --train_size 1 --data_set cifar10  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='cifar10', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=1.0, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
10
10
Training size is:  40000
Number of epochs is:  2
Number of iterations per round is:  1250
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7f7c930eff70> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 14:40:06.836546: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 14:40:09.924232: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-07 14:40:13.457385: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8201
2023-03-07 14:40:17.242560: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 14:40:17.243510: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 14:40:17.243528: W tensorflow/stream_executor/gpu/asm_compiler.cc:80] Couldn't get ptxas version string: INTERNAL: Couldn't invoke ptxas --version
2023-03-07 14:40:17.243888: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 14:40:17.243919: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] INTERNAL: Failed to launch ptxas
Relying on driver to perform ptx compilation. 
Modify $PATH to customize ptxas location.
This message will be only logged once.
    519.8891645078329 examples per second
    26065.651758093994 examples per second
    26208.49255654885 examples per second
    26011.307860997123 examples per second
    26012.31764577542 examples per second
    26293.767595659392 examples per second
-----------------------------------
Temperature has increased to  200.0
    26313.507983271633 examples per second
    26225.922237241466 examples per second
    26254.13263079816 examples per second
    26155.332842646876 examples per second
    25959.391687114297 examples per second
    26506.895881934037 examples per second
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 1 --data_set cifar10  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='cifar10', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=1.0, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
10
10
Training size is:  40000
Number of epochs is:  50
Number of iterations per round is:  31250
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7f292338af70> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 14:46:43.493347: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 14:46:44.015414: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-07 14:46:44.940252: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8201
2023-03-07 14:46:45.316717: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 14:46:45.317496: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 14:46:45.317514: W tensorflow/stream_executor/gpu/asm_compiler.cc:80] Couldn't get ptxas version string: INTERNAL: Couldn't invoke ptxas --version
2023-03-07 14:46:45.317790: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 14:46:45.317826: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] INTERNAL: Failed to launch ptxas
Relying on driver to perform ptx compilation. 
Modify $PATH to customize ptxas location.
This message will be only logged once.
Step 0:    (2023-03-07 14:46:47.273980)
    training nat accuracy 10.94
    validation nat accuracy 9.93
    Nat Xent 2.309
    Regularizer 0.0
theta:  0.1
best
Step 100:    (2023-03-07 14:46:55.317113)
    training nat accuracy 35.94
    validation nat accuracy 34.13
    Nat Xent 1.722
    Regularizer 0.0
theta:  0.1
best
^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^CTraceback (most recent call last):
  File "train.py", line 199, in <module>
    dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment, train_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_model.py", line 83, in update_dict
    dict_exp[w_vars[i] + '_nonzero'][experiment] = sum(W.reshape(-1) != 0) / W.reshape(-1).shape[0]
KeyboardInterrupt
^C^C^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 1 --data_set mnist  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='mnist', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=1.0, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
Traceback (most recent call last):
  File "train.py", line 67, in <module>
    data, data_shape = input_data.load_data_set(training_size = train_size, validation_size= val_size, data_set=data_set, seed=seed)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/input_data.py", line 203, in load_data_set
    X_train, X_left, y_train, y_left = train_test_split(X_train, y_train, test_size=(X_train.shape[0] - n), random_state=seed)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/sklearn/model_selection/_split.py", line 2420, in train_test_split
    n_train, n_test = _validate_shuffle_split(
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/sklearn/model_selection/_split.py", line 2043, in _validate_shuffle_split
    raise ValueError(
ValueError: test_size=0 should be either positive and smaller than the number of samples 60000 or a float in the (0, 1) range
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 1 --data_set mnist  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='mnist', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=1.0, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
10
10
Training size is:  48000
Number of epochs is:  50
Number of iterations per round is:  37500
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7fa2e93e1040> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 15:19:48.381162: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 15:19:48.835409: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-07 15:19:49.639982: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8201
2023-03-07 15:19:50.054643: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:19:50.056102: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:19:50.056156: W tensorflow/stream_executor/gpu/asm_compiler.cc:80] Couldn't get ptxas version string: INTERNAL: Couldn't invoke ptxas --version
2023-03-07 15:19:50.056558: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:19:50.056597: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] INTERNAL: Failed to launch ptxas
Relying on driver to perform ptx compilation. 
Modify $PATH to customize ptxas location.
This message will be only logged once.
Step 0:    (2023-03-07 15:19:51.753156)
    training nat accuracy 6.25
    validation nat accuracy 3.858
    Nat Xent 2.327
    Regularizer 0.0
theta:  0.1
best
^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^CTraceback (most recent call last):
  File "train.py", line 199, in <module>
    dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment, train_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_model.py", line 83, in update_dict
    dict_exp[w_vars[i] + '_nonzero'][experiment] = sum(W.reshape(-1) != 0) / W.reshape(-1).shape[0]
KeyboardInterrupt
^C^C^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set mnist  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='mnist', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=0.8, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
10
10
Training size is:  38400
Number of epochs is:  50
Number of iterations per round is:  30000
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7f3b8b5b4280> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 15:20:09.661981: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 15:20:10.110102: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-07 15:20:10.893026: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8201
2023-03-07 15:20:11.286918: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:20:11.287385: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:20:11.287403: W tensorflow/stream_executor/gpu/asm_compiler.cc:80] Couldn't get ptxas version string: INTERNAL: Couldn't invoke ptxas --version
2023-03-07 15:20:11.287707: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:20:11.287744: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] INTERNAL: Failed to launch ptxas
Relying on driver to perform ptx compilation. 
Modify $PATH to customize ptxas location.
This message will be only logged once.
Step 0:    (2023-03-07 15:20:12.911618)
    training nat accuracy 3.125
    validation nat accuracy 3.958
    Nat Xent 2.336
    Regularizer 0.0
theta:  0.1
best
^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^CTraceback (most recent call last):
  File "train.py", line 199, in <module>
    dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment, train_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_model.py", line 83, in update_dict
    dict_exp[w_vars[i] + '_nonzero'][experiment] = sum(W.reshape(-1) != 0) / W.reshape(-1).shape[0]
KeyboardInterrupt
^C^C^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set cifar10  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='cifar10', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=0.8, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
10
10
Training size is:  32000
Number of epochs is:  50
Number of iterations per round is:  25000
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7f390f189f70> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 15:20:35.062134: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 15:20:35.513474: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-07 15:20:36.388059: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8201
2023-03-07 15:20:36.771353: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:20:36.772027: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:20:36.772048: W tensorflow/stream_executor/gpu/asm_compiler.cc:80] Couldn't get ptxas version string: INTERNAL: Couldn't invoke ptxas --version
2023-03-07 15:20:36.772852: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:20:36.772967: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] INTERNAL: Failed to launch ptxas
Relying on driver to perform ptx compilation. 
Modify $PATH to customize ptxas location.
This message will be only logged once.
Step 0:    (2023-03-07 15:20:38.676652)
    training nat accuracy 9.375
    validation nat accuracy 10.23
    Nat Xent 2.366
    Regularizer 0.0
theta:  0.1
best
^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C^C



Traceback (most recent call last):
  File "train.py", line 199, in <module>
    dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment, train_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_model.py", line 83, in update_dict
    dict_exp[w_vars[i] + '_nonzero'][experiment] = sum(W.reshape(-1) != 0) / W.reshape(-1).shape[0]
KeyboardInterrupt

kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set hola  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='hola', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=0.8, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
Traceback (most recent call last):
  File "train.py", line 67, in <module>
    data, data_shape = input_data.load_data_set(training_size = train_size, validation_size= val_size, data_set=data_set, seed=seed)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/input_data.py", line 183, in load_data_set
    assert int(data_set) in range(len(UCI)), "Unknown data set!"
ValueError: invalid literal for int() with base 10: 'hola'
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim config.json 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set 0  --network_type ALEX --exp_id -1
Traceback (most recent call last):
  File "train.py", line 17, in <module>
    import input_data
  File "/home/gridsan/kimvc/final_HDL/HDL/src/input_data.py", line 11, in <module>
    config = json.load(config_file)
NameError: name 'json' is not defined
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set 0  --network_type ALEX --exp_id -1
Traceback (most recent call last):
  File "train.py", line 17, in <module>
    import input_data
  File "/home/gridsan/kimvc/final_HDL/HDL/src/input_data.py", line 12, in <module>
    config = json.load(config_file)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/json/__init__.py", line 293, in load
    return loads(fp.read(),
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/json/__init__.py", line 357, in loads
    return _default_decoder.decode(s)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/json/decoder.py", line 353, in raw_decode
    obj, end = self.scan_once(s, idx)
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 30 column 10 (char 732)
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim config.json 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set 0  --network_type ALEX --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='0', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='ALEX', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=0.8, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
2
2
Training size is:  1297
Number of epochs is:  150
Number of iterations per round is:  3000
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7fdf6f1fcee0> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 15:54:36.113187: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 15:54:36.598364: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
2023-03-07 15:54:37.285712: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8201
2023-03-07 15:54:37.644009: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:54:37.644667: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:54:37.644684: W tensorflow/stream_executor/gpu/asm_compiler.cc:80] Couldn't get ptxas version string: INTERNAL: Couldn't invoke ptxas --version
2023-03-07 15:54:37.645455: I tensorflow/core/platform/default/subprocess.cc:304] Start cannot spawn child process: No such file or directory
2023-03-07 15:54:37.645585: W tensorflow/stream_executor/gpu/redzone_allocator.cc:314] INTERNAL: Failed to launch ptxas
Relying on driver to perform ptx compilation. 
Modify $PATH to customize ptxas location.
This message will be only logged once.
2023-03-07 15:54:38.147817: W tensorflow/core/framework/op_kernel.cc:1733] INVALID_ARGUMENT: required broadcastable shapes
Traceback (most recent call last):
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1377, in _do_call
    return fn(*args)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1360, in _run_fn
    return self._call_tf_sessionrun(options, feed_dict, fetch_list,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1453, in _call_tf_sessionrun
    return tf_session.TF_SessionRun_wrapper(self._session, options, feed_dict,
tensorflow.python.framework.errors_impl.InvalidArgumentError: 2 root error(s) found.
  (0) INVALID_ARGUMENT: required broadcastable shapes
	 [[{{node Equal}}]]
	 [[Mean_4/_9]]
  (1) INVALID_ARGUMENT: required broadcastable shapes
	 [[{{node Equal}}]]
0 successful operations.
0 derived errors ignored.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "train.py", line 185, in <module>
    utils_print.print_metrics(sess, model, nat_dict, val_dict, test_dict, train_step, args, summary_writer, experiment, global_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_print.py", line 20, in print_metrics
    nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 967, in run
    result = self._run(None, fetches, feed_dict, options_ptr,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1190, in _run
    results = self._do_run(handle, final_targets, final_fetches,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1370, in _do_run
    return self._do_call(_run_fn, feeds, fetches, targets, options,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1396, in _do_call
    raise type(e)(node_def, op, message)  # pylint: disable=no-value-for-parameter
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:

Detected at node 'Equal' defined at (most recent call last):
    File "train.py", line 131, in <module>
      model = network_module.Model(num_classes, batch_size, network_size, pool_size, subset_ratio, num_features, dropout, l2, l0, rho, data_shape, ticket, stored_weights)
    File "./Networks/CNN_model.py", line 192, in __init__
      correct_prediction = tf.equal(self.y_pred, self.y_input)
Node: 'Equal'
Detected at node 'Equal' defined at (most recent call last):
    File "train.py", line 131, in <module>
      model = network_module.Model(num_classes, batch_size, network_size, pool_size, subset_ratio, num_features, dropout, l2, l0, rho, data_shape, ticket, stored_weights)
    File "./Networks/CNN_model.py", line 192, in __init__
      correct_prediction = tf.equal(self.y_pred, self.y_input)
Node: 'Equal'
2 root error(s) found.
  (0) INVALID_ARGUMENT: required broadcastable shapes
	 [[{{node Equal}}]]
	 [[Mean_4/_9]]
  (1) INVALID_ARGUMENT: required broadcastable shapes
	 [[{{node Equal}}]]
0 successful operations.
0 derived errors ignored.

Original stack trace for 'Equal':
  File "train.py", line 131, in <module>
    model = network_module.Model(num_classes, batch_size, network_size, pool_size, subset_ratio, num_features, dropout, l2, l0, rho, data_shape, ticket, stored_weights)
  File "./Networks/CNN_model.py", line 192, in __init__
    correct_prediction = tf.equal(self.y_pred, self.y_input)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/traceback_utils.py", line 150, in error_handler
    return fn(*args, **kwargs)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py", line 1082, in op_dispatch_handler
    return dispatch_target(*args, **kwargs)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/ops/math_ops.py", line 1922, in equal
    return gen_math_ops.equal(x, y, name=name)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/ops/gen_math_ops.py", line 3319, in equal
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/framework/op_def_library.py", line 740, in _apply_op_helper
    op = g._create_op_internal(op_type_name, inputs, dtypes=None,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 3776, in _create_op_internal
    ret = Operation(
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 2175, in __init__
    self._traceback = tf_stack.extract_stack_for_node(self._c_op)

kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 0.8 --data_set 0  --network_type MLP --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='0', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='MLP', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=0.8, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
2
2
Training size is:  1297
Number of epochs is:  150
Number of iterations per round is:  3000
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7f862472fc10> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 15:55:57.683964: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 15:55:58.164229: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
Step 0:    (2023-03-07 15:55:59.021111)
    training nat accuracy 100.0
    validation nat accuracy 93.52
    Nat Xent 0.24
    Regularizer 0.0
theta:  0.1
best
-----------------------------------
Temperature has increased to  1.0361989654150687
-----------------------------------
Temperature has increased to  1.0737082959272588
-----------------------------------
Temperature has increased to  1.112575425397402
-----------------------------------
Temperature has increased to  1.152849504743018
-----------------------------------
Temperature has increased to  1.1945814640939896
Step 100:    (2023-03-07 15:55:59.227109)
    training nat accuracy 90.62
    validation nat accuracy 93.52
    Nat Xent 0.3114
    Regularizer 0.0
theta:  0.1
best
    65013.633784007376 examples per second
-----------------------------------
Temperature has increased to  1.23782407719821
-----------------------------------
Temperature has increased to  1.2826320281586474
-----------------------------------
Temperature has increased to  1.3290619805862218
-----------------------------------
Temperature has increased to  1.3771726492559453
-----------------------------------
Temperature has increased to  1.4270248743569398
Step 200:    (2023-03-07 15:55:59.328582)
    training nat accuracy 98.44
    validation nat accuracy 93.52
    Nat Xent 0.1154
    Regularizer 0.0
theta:  0.1
best
    115212.27702788354 examples per second
-----------------------------------
Temperature has increased to  1.4786816984302296
-----------------------------------
Temperature has increased to  1.5322084460916006
-----------------------------------
Temperature has increased to  1.5876728066403467
-----------------------------------
Temperature has increased to  1.6451449196583656
-----------------------------------
Temperature has increased to  1.704697463707855
Step 300:    (2023-03-07 15:55:59.440209)
    training nat accuracy 93.75
    validation nat accuracy 93.52
    Nat Xent 0.2054
    Regularizer 0.0
theta:  0.1
best
    100066.42845981664 examples per second
-----------------------------------
Temperature has increased to  1.766405748239771
-----------------------------------
Temperature has increased to  1.830347808829281
-----------------------------------
Temperature has increased to  1.896604505858639
-----------------------------------
Temperature has increased to  1.9652596267722793
-----------------------------------
Temperature has increased to  2.0363999920334397
Step 400:    (2023-03-07 15:55:59.552651)
    training nat accuracy 98.44
    validation nat accuracy 93.52
    Nat Xent 0.1023
    Regularizer 0.0
theta:  0.1
best
    107771.40675620048 examples per second
-----------------------------------
Temperature has increased to  2.1101155649163044
-----------------------------------
Temperature has increased to  2.186499565272508
-----------------------------------
Temperature has increased to  2.2656485874158703
-----------------------------------
Temperature has increased to  2.347662722274437
-----------------------------------
Temperature has increased to  2.4326456839642954
Step 500:    (2023-03-07 15:55:59.663198)
    training nat accuracy 96.88
    validation nat accuracy 93.52
    Nat Xent 0.1544
    Regularizer 0.0
theta:  0.1
best
    101799.89455660841 examples per second
-----------------------------------
Temperature has increased to  2.520704940945235
-----------------------------------
Temperature has increased to  2.6119518519241045
-----------------------------------
Temperature has increased to  2.7065018066777298
-----------------------------------
Temperature has increased to  2.804474371973478
-----------------------------------
Temperature has increased to  2.905993442771993
Step 600:    (2023-03-07 15:55:59.769918)
    training nat accuracy 96.88
    validation nat accuracy 93.52
    Nat Xent 0.1296
    Regularizer 0.0
theta:  0.1
best
    110756.41954100125 examples per second
-----------------------------------
Temperature has increased to  3.0111873989033127
-----------------------------------
Temperature has increased to  3.1201892674145046
-----------------------------------
Temperature has increased to  3.233136890794111
-----------------------------------
Temperature has increased to  3.35017310128615
-----------------------------------
Temperature has increased to  3.471445901514101
Step 700:    (2023-03-07 15:55:59.873306)
    training nat accuracy 93.75
    validation nat accuracy 93.52
    Nat Xent 0.1898
    Regularizer 0.0
theta:  0.1
best
    112875.5228607895 examples per second
-----------------------------------
Temperature has increased to  3.5971086516432917
-----------------------------------
Temperature has increased to  3.7273202633183717
-----------------------------------
Temperature has increased to  3.862245400621118
-----------------------------------
Temperature has increased to  4.00205468830271
-----------------------------------
Temperature has increased to  4.146924927553794
Step 800:    (2023-03-07 15:55:59.982865)
    training nat accuracy 95.31
    validation nat accuracy 93.52
    Nat Xent 0.1789
    Regularizer 0.0
theta:  0.1
best
    105181.62540154372 examples per second
-----------------------------------
Temperature has increased to  4.2970393195852
-----------------------------------
Temperature has increased to  4.452587697302055
-----------------------------------
Temperature has increased to  4.613766765364252
-----------------------------------
Temperature has increased to  4.780780348936866
-----------------------------------
Temperature has increased to  4.9538396514450715
Step 900:    (2023-03-07 15:56:00.098269)
    training nat accuracy 87.5
    validation nat accuracy 93.52
    Nat Xent 0.3182
    Regularizer 0.0
theta:  0.1
best
    94852.66798952776 examples per second
-----------------------------------
Temperature has increased to  5.133163521659528
-----------------------------------
Temperature has increased to  5.318978730449973
-----------------------------------
Temperature has increased to  5.5115202575570175
-----------------------------------
Temperature has increased to  5.711031588744775
-----------------------------------
Temperature has increased to  5.917765023710112
Step 1000:    (2023-03-07 15:56:00.214972)
    training nat accuracy 95.31
    validation nat accuracy 93.52
    Nat Xent 0.1565
    Regularizer 0.0
theta:  0.1
best
    95248.19728263986 examples per second
-----------------------------------
Temperature has increased to  6.131981995137898
-----------------------------------
Temperature has increased to  6.353953399305719
-----------------------------------
Temperature has increased to  6.583959938656145
-----------------------------------
Temperature has increased to  6.822292476769757
-----------------------------------
Temperature has increased to  7.069252406187829
Step 1100:    (2023-03-07 15:56:00.320235)
    training nat accuracy 95.31
    validation nat accuracy 93.52
    Nat Xent 0.1683
    Regularizer 0.0
theta:  0.1
best
    111992.54770620217 examples per second
-----------------------------------
Temperature has increased to  7.325152029549813
-----------------------------------
Temperature has increased to  7.590314954527607
-----------------------------------
Temperature has increased to  7.865076503056032
-----------------------------------
Temperature has increased to  8.149784135377027
-----------------------------------
Temperature has increased to  8.444797889433817
^CTraceback (most recent call last):
  File "train.py", line 185, in <module>
    utils_print.print_metrics(sess, model, nat_dict, val_dict, test_dict, train_step, args, summary_writer, experiment, global_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_print.py", line 25, in print_metrics
    robust_xent = sess.run(model.robust_xent, feed_dict=nat_dict)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 967, in run
    result = self._run(None, fetches, feed_dict, options_ptr,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1190, in _run
    results = self._do_run(handle, final_targets, final_fetches,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1370, in _do_run
    return self._do_call(_run_fn, feeds, fetches, targets, options,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1377, in _do_call
    return fn(*args)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1360, in _run_fn
    return self._call_tf_sessionrun(options, feed_dict, fetch_list,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1453, in _call_tf_sessionrun
    return tf_session.TF_SessionRun_wrapper(self._session, options, feed_dict,
KeyboardInterrupt
^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ python train.py --train_size 1 --data_set 0  --network_type MLP --exp_id -1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/compat/v2_compat.py:107: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
Instructions for updating:
non-resource variables are not supported in the long term
Namespace(batch_range=[64], data_set='0', dropout=1, exp_id=-1, is_stable=False, l0=0, l2=0, lr=0.0001, network_type='MLP', reg_stability=0, rho=0, robust_test=[1e-05, 0.0001, 0.001, 0.01, 0.1], stab_ratio_range=[0.8], train_size=1.0, val_size=0.2)
Batch Size: 64  ; stability subset ratio: 0.8  ; dropout value: 1
2
2
Training size is:  1622
Number of epochs is:  120
Number of iterations per round is:  3000
__________________________________________________________
__________________________________________________________
Starting round # 0  for experiment # 0
---------------------------------
Temperature initialized to  1
WARNING:tensorflow:From /state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/util/dispatch.py:1082: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
WARNING:tensorflow:AutoGraph could not transform <function Model.__init__.<locals>.<lambda> at 0x7fdb8f1c8c10> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: 'NoneType' object has no attribute '__dict__'
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
2023-03-07 15:56:13.997227: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-03-07 15:56:14.472823: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1525] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 30989 MB memory:  -> device: 0, name: Tesla V100-PCIE-32GB, pci bus id: 0000:86:00.0, compute capability: 7.0
Step 0:    (2023-03-07 15:56:15.320779)
    training nat accuracy 90.62
    validation nat accuracy 93.09
    Nat Xent 0.3266
    Regularizer 0.0
theta:  0.1
best
-----------------------------------
Temperature has increased to  1.0455297296988268
-----------------------------------
Temperature has increased to  1.093132415684102
-----------------------------------
Temperature has increased to  1.1429024390952247
-----------------------------------
Temperature has increased to  1.1949384782193602
Step 100:    (2023-03-07 15:56:15.513729)
    training nat accuracy 89.06
    validation nat accuracy 93.33
    Nat Xent 0.3844
    Regularizer 0.0
theta:  0.1
best
    71188.73488947512 examples per second
-----------------------------------
Temperature has increased to  1.249343704139415
-----------------------------------
Temperature has increased to  1.3062259852898137
-----------------------------------
Temperature has increased to  1.3656981013256426
-----------------------------------
Temperature has increased to  1.4278779667292
Step 200:    (2023-03-07 15:56:15.609382)
    training nat accuracy 90.62
    validation nat accuracy 93.33
    Nat Xent 0.3187
    Regularizer 0.0
theta:  0.1
best
    128299.25337182042 examples per second
-----------------------------------
Temperature has increased to  1.492888864597291
-----------------------------------
Temperature has increased to  1.5608596910727943
-----------------------------------
Temperature has increased to  1.631925210905133
-----------------------------------
Temperature has increased to  1.7062263246463447
Step 300:    (2023-03-07 15:56:15.702960)
    training nat accuracy 90.62
    validation nat accuracy 93.33
    Nat Xent 0.3063
    Regularizer 0.0
theta:  0.1
best
    128456.78071649773 examples per second
-----------------------------------
Temperature has increased to  1.7839103480125156
-----------------------------------
Temperature has increased to  1.8651313039644655
-----------------------------------
Temperature has increased to  1.950050228086788
-----------------------------------
Temperature has increased to  2.038835487870715
Step 400:    (2023-03-07 15:56:15.801680)
    training nat accuracy 96.88
    validation nat accuracy 93.33
    Nat Xent 0.1593
    Regularizer 0.0
theta:  0.1
best
    120074.06546256553 examples per second
-----------------------------------
Temperature has increased to  2.1316631165338444
-----------------------------------
Temperature has increased to  2.2287171620385893
-----------------------------------
Temperature has increased to  2.3301900520013428
-----------------------------------
Temperature has increased to  2.436282975215859
Step 500:    (2023-03-07 15:56:15.903337)
    training nat accuracy 89.06
    validation nat accuracy 93.33
    Nat Xent 0.3301
    Regularizer 0.0
theta:  0.1
best
    120686.26599855983 examples per second
-----------------------------------
Temperature has increased to  2.547206280547291
-----------------------------------
Temperature has increased to  2.663179893987763
-----------------------------------
Temperature has increased to  2.784433754700376
-----------------------------------
Temperature has increased to  2.9112082709161737
^CTraceback (most recent call last):
  File "train.py", line 185, in <module>
    utils_print.print_metrics(sess, model, nat_dict, val_dict, test_dict, train_step, args, summary_writer, experiment, global_step)
  File "/home/gridsan/kimvc/final_HDL/HDL/src/../utils/utils_print.py", line 27, in print_metrics
    stable_var = sess.run(getattr(model, stable_var), feed_dict=nat_dict)
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 967, in run
    result = self._run(None, fetches, feed_dict, options_ptr,
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/client/session.py", line 1163, in _run
    not subfeed_t.get_shape().is_compatible_with(np_val.shape)):
  File "/state/partition1/llgrid/pkg/anaconda/anaconda3-2022a/lib/python3.8/site-packages/tensorflow/python/framework/tensor_shape.py", line 1149, in is_compatible_with
    for x_dim, y_dim in zip(self._dims, other.dims):
KeyboardInterrupt
^C^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ ^C
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim config.json 
kimvc@d-14-5-2:~/final_HDL/HDL/src$ vim input_data.py 

      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

def load_data_set(training_size, validation_size, data_set, seed=None, reshape=True, dtype=dtypes.float32, standardize=False):
  if data_set == "cifar10":
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    num_features = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]

  elif data_set == "fashion_mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    if not reshape:
        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
    num_features = X_train.shape[1]*X_train.shape[2]

  elif data_set == "mnist":
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    if not reshape:
        X_train = X_train[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
    num_features = X_train.shape[1]*X_train.shape[2]

  else:
    assert data_set in UCI, "Unknown data set!"
    uci_name = UCI[data_set]
    X = np.genfromtxt("../UCI/" + str(uci_name) + "_X.csv", delimiter=',')
    Y = np.genfromtxt("../UCI/" + str(uci_name) + "_Y.csv", delimiter=',')

    K = len(np.unique((Y)))
    labels = np.unique(Y)
    dict_labels = {labels[k]:k for k in range(K)}
    for k in range(Y.shape[0]):
        Y[k] = dict_labels[Y[k]]
    num_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=validation_size, random_state=0)

  n = int(X_train.shape[0]*training_size)
  m = int(n*validation_size)

  if training_size != 1:
      X_train, X_left, y_train, y_left = train_test_split(X_train, y_train, test_size=(X_train.shape[0] - n), random_state=seed)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=m, random_state=seed)

  if standardize:
      scaler = preprocessing.StandardScaler().fit(X_train)
      X_train = scaler.transform(X_train)
      X_val = scaler.transform(X_val)
      X_test = scaler.transform(X_test)

  options = dict(dtype=dtype, reshape=reshape, num_features=num_features, seed=seed)

  train = _DataSet(X_train, y_train, **options )
  validation = _DataSet(X_val, y_val, **options)
  test = _DataSet(X_test, y_test, **options)

  return _Datasets(train=train, validation=validation, test=test), X_train.shape
