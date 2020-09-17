"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import genfromtxt

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import data
from NN_model import Model

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch-size", type=int, default=64,
                            help="batch size")

parser.add_argument("--num-subsets", type=int, default=5,
                            help="number of subsets")

parser.add_argument("--subset-ratio", type=float, default=0.8,
                            help="how much subset ratio for training")

with open('config.json') as config_file:
    config = json.load(config_file)


args = parser.parse_args()
print(args)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
#batch_size = config['batch_size']
batch_size = args.batch_size
#num_subsets= config['num_subsets']
num_subsets = args.num_subsets
#subset_ratio = config['subset_ratio']
subset_ratio = args.subset_ratio
subset_batch_size = int(batch_size/num_subsets)
initial_learning_rate = config['initial_learning_rate']
eta = config['constant_learning_rate']
learning_rate = tf.train.exponential_decay(initial_learning_rate, 0, 5, 0.85, staircase=True)

# Setting up the data and the model
mnist = data.load_mnist_data_set(num_subsets, subset_ratio, validation_size=10000)
global_step = tf.Variable(1, name="global_step")
model = Model(num_subsets, subset_batch_size)

# Setting up the optimizer

#CONSTANC STEP SIZE
#optimizer = tf.train.AdamOptimizer(eta).minimize(model.max_xent, global_step=global_step)
#DECREASING STEP SIZE
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.xent, global_step=global_step)

avg_test_acc = 0
test_accs = {}
num_experiments = config['num_experiments']

for experiment in range(num_experiments):
  # Setting up the Tensorboard and checkpoint outputs
  model_dir = config['model_dir'] + str(datetime.now())
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  with tf.Session() as sess:
    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(model_dir + "/Xent")
    summary_writer1 = tf.summary.FileWriter(model_dir+ "/Max_Xent")
    summary_writer2 = tf.summary.FileWriter(model_dir+ "/Accuracy")
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for ii in range(max_num_training_steps):
      x_batch, y_batch = mnist.train.next_batch(batch_size)

      nat_dict = {model.x_input: x_batch,
                  model.y_input: y_batch}

      # Output
      if ii % num_output_steps == 0:
        nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        nat_xent = sess.run(model.xent, feed_dict=nat_dict)
        max_xent = sess.run(model.max_xent, feed_dict=nat_dict)
        print('Step {}:    ({})'.format(ii, datetime.now()))
        print('    training nat accuracy {:.4}'.format(nat_acc * 100))
        print('    Nat Xent {:.4}'.format(nat_xent/batch_size))
        print('    Max Xent {:.4}'.format(max_xent/subset_batch_size))

        #Tensorboard Summaries
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Xent', simple_value= nat_xent/batch_size),])
        summary1 = tf.Summary(value=[
            tf.Summary.Value(tag='Xent', simple_value= max_xent/subset_batch_size),])
        summary2 = tf.Summary(value=[
            tf.Summary.Value(tag='Accuracy', simple_value= nat_acc*100)])
        summary_writer.add_summary(summary, global_step.eval(sess))
        summary_writer1.add_summary(summary1, global_step.eval(sess))
        summary_writer2.add_summary(summary2, global_step.eval(sess))

        if ii != 0:
          print('    {} examples per second'.format(
              num_output_steps * batch_size / training_time))
          training_time = 0.0
      

      # Actual training step
      start = timer()
      sess.run(optimizer, feed_dict=nat_dict)
      end = timer()
      training_time += end - start


    #Output test results
    test_dict = {model.x_input: mnist.test.all_images,
                model.y_input: mnist.test.all_labels} 
    test_acc= sess.run(model.accuracy, feed_dict=test_dict)
    test_accs[experiment] = test_acc  * 100
    avg_test_acc += test_acc

avg_test_acc  = avg_test_acc/num_experiments
print('  average testing accuracy {:.4}'.format(avg_test_acc  * 100))
print('  individual accuracies: \n', test_accs)
print('  Standard deviation {:.2}'.format(np.array([float(test_accs[k]) for k in test_accs]).std()))



