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
import input_data
import csv
import itertools
from utils import total_gini

import argparse



parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--batch_range", type=int, nargs='+', default=[64],
                            help="batch range")

parser.add_argument("--ratio_range", type=float, nargs='+', default=[0.8],
                            help="ratio range")

parser.add_argument("--model", "-m", type=str, required=True,
    choices=["ff", "cnn"],
                            help="model")

parser.add_argument("--stable", action="store_true",
                            help="stable version")

parser.add_argument("--dropout", type=float, default=1,
                            help="dropout rate, 1 is no dropout, 0 is all set to 0")

parser.add_argument("--l2", type=float, default=0,
                            help="l2 regularization rate")

parser.add_argument("--l0", type=float, default=0,
                            help="l0 regularization rate")

parser.add_argument("--reg_stability", type=float, default=0,
                            help="reg stability regularization rate")

parser.add_argument("--num_subsets", type=int, default=1,
                            help="number of subsets for Monte Carlo")

parser.add_argument("--l1_size", type=int, default=512,
                            help="number of nodes in the first layer, 784 -> l1_size")

parser.add_argument("--l2_size", type=int, default=256,
                            help="number of nodes in the first layer, l1_size -> l2_size")

parser.add_argument("--cnn_size", type=int, default=32,
                            help="number of filters in the cnn layers for cnn")

parser.add_argument("--fc_size", type=int, default=128,
                            help="number of nodes in the dense layer for cnn")

parser.add_argument("--data_set", type=str, default="mnist",
                            help="number of subsets")

parser.add_argument("--MC", action="store_true",
                            help="Monte Carlo version")

parser.add_argument("--train_size", type=float, default=1,
                            help="training percent of the data")

parser.add_argument("--val_size", type=float, default=0.25,
                            help="validation percent of the data e.g., 0.25 means 0.25*traning size")

with open('config.json') as config_file:
    config = json.load(config_file)


args = parser.parse_args()
print(args)
model_type = args.model
if model_type == "ff":
    from NN_model import Model
elif model_type == "cnn":
    from CNN_model import Model
    assert(tf.keras.backend.image_data_format() == "channels_last")
# Setting up training parameters
seed = config['random_seed']
tf.set_random_seed(seed)
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
testing_size = config['testing_size']
data_set = args.data_set
batch_range = args.batch_range
ratio_range = args.ratio_range
num_subsets = args.num_subsets
stable = args.stable
MC = args.MC
dropout = args.dropout
l2 = args.l2
initial_learning_rate = config['initial_learning_rate']
eta = config['constant_learning_rate']
learning_rate = tf.train.exponential_decay(initial_learning_rate,
 0, 5, 0.85, staircase=True)

global_step = tf.Variable(1, name="global_step")


for batch_size, subset_ratio in itertools.product(batch_range, ratio_range): #Parameters chosen with validation
  print(batch_size, subset_ratio, dropout)

  #Setting up the data and the model
  if model_type == "ff":
      data = input_data.load_data_set(training_size = args.train_size, validation_size=args.val_size, data_set=data_set, seed=seed)
      num_features = data.train.images.shape[1]
      model = Model(num_subsets, batch_size, args.l1_size, args.l2_size, subset_ratio, num_features, dropout, l2, args.l0, args.reg_stability)
      var_list = [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]
  elif model_type == "cnn":
      data = input_data.load_data_set(training_size = args.train_size, validation_size=args.val_size, data_set=data_set, reshape=False, seed=seed)
      print(data.train.images.shape)
      pixels_x = data.train.images.shape[1]
      pixels_y = data.train.images.shape[2]
      num_channels = data.train.images.shape[3]
      if stable and (not MC):
          theta = True
      else:
          theta = False
      model = Model(num_subsets, batch_size, args.cnn_size, args.fc_size, subset_ratio, pixels_x,pixels_y,num_channels, dropout, l2, theta)
      
  if MC:
    max_loss = model.MC_xent
  else:
    max_loss = model.dual_xent

  #Setting up data for testing and validation
  val_dict = {model.x_input: data.validation.images,
                  model.y_input: data.validation.labels.reshape(-1)}
  test_dict = {model.x_input: data.test.images[:testing_size],
                  model.y_input: data.test.labels[:testing_size].reshape(-1)}

  # Setting up the optimizer
  if model_type == "ff":
      if stable:
    
          if MC:
    
            if l2+args.l0 > 0:
                optimizer = tf.train.AdamOptimizer(eta).minimize(max_loss + model.regularizer, global_step=global_step, var_list=var_list)
            else:
                #DECAY STEP SIZE STEP SIZE
                optimizer = tf.train.AdamOptimizer(eta).minimize(max_loss, global_step=global_step, var_list=var_list)
                #DECREASING STEP SIZE
                #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.xent, global_step=global_step)
    
          else:
    
            if l2 > 0:
                optimizer = tf.train.AdamOptimizer(eta).minimize(max_loss + model.regularizer, global_step=global_step)
            else:
                #DECAY STEP SIZE STEP SIZE
                optimizer = tf.train.AdamOptimizer(eta).minimize(max_loss, global_step=global_step)
                #DECREASING STEP SIZE
                #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.xent, global_step=global_step)
    
    
    
      else:
          #CONSTANC STEP SIZE
          optimizer = tf.train.AdamOptimizer(eta).minimize(model.xent + model.regularizer, global_step=global_step, var_list=var_list)
          #DECREASING STEP SIZE
          #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.xent, global_step=global_step, var_list=var_list)
  elif model_type == "cnn":
    if stable:
        optimizer = tf.train.AdamOptimizer(eta).minimize(max_loss, global_step=global_step)
    else:
        optimizer = tf.train.AdamOptimizer(eta).minimize(model.xent, global_step=global_step)
  #DECREASING STEP SIZE
  #optimizer = tf.train.AdamOptimizer(learning_rate).minimize(model.xent, global_step=global_step, var_list=var_list)      

  #Initializing loop variables.
  avg_test_acc = 0
  test_accs = {}
  thetas = {}
  iterations = {}
  num_experiments = config['num_experiments']
  if model_type == "ff":
      logits_acc = np.zeros((config['num_experiments'], 10000, 10))
      W1_acc = np.zeros((config['num_experiments'], num_features*args.l1_size))
      W2_acc = np.zeros((config['num_experiments'], args.l1_size*args.l2_size))
      W3_acc = np.zeros((config['num_experiments'], args.l2_size * 10))
      #TODO: replace 10000 by the actual size of test set -- solved?
      preds = np.zeros((config['num_experiments'], data.test.images.shape[0]))
    
      for experiment in range(num_experiments):
        print("Experiment", experiment)
    
        # Setting up the Tensorboard and checkpoint outputs
        model_dir = config['model_dir'] + str(datetime.now())
        if not os.path.exists(model_dir):
          os.makedirs(model_dir)
    
        with tf.Session() as sess:
          # Initialize the summary writer, global variables, and our time counter.
          #summary_writer = tf.summary.FileWriter(model_dir + "/Xent")
          #summary_writer1 = tf.summary.FileWriter(model_dir+ "/Max_Xent")
          #summary_writer2 = tf.summary.FileWriter(model_dir+ "/Accuracy")
          #summary_writer3 = tf.summary.FileWriter(model_dir+ "/Test_Accuracy")
          sess.run(tf.global_variables_initializer())
          training_time = 0.0
    
          # Main training loop
          best_val_acc = 0
          test_acc = 0
          num_iters = 0
          for ii in range(max_num_training_steps):
            x_batch, y_batch = data.train.next_batch(batch_size)
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
    
            # Output
            if ii % num_output_steps == 0:
              nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
              val_acc = sess.run(model.accuracy, feed_dict=val_dict)
              nat_xent = sess.run(model.xent, feed_dict=nat_dict)
              dual_xent = sess.run(model.dual_xent, feed_dict=nat_dict)
              MC_xent = sess.run(model.MC_xent, feed_dict=nat_dict)
              regularizer = sess.run(model.regularizer, feed_dict=nat_dict)
              print('Step {}:    ({})'.format(ii, datetime.now()))
              print('    training nat accuracy {:.4}'.format(nat_acc * 100))
              print('    validation nat accuracy {:.4}'.format(val_acc * 100))
              print('    Nat Xent {:.4}'.format(nat_xent))
              print('    Max Xent upper bound with dual {:.4}'.format(dual_xent))
              print('    Max Xent lower bound with Monte Carlo {:.4}'.format(MC_xent))
              print('    Regularizer', regularizer)
              if args.l0 > 0:
                print('    W1_masked features', sum(sess.run(model.W1_masked).reshape(-1) > 0))
                print('    W2_masked features', sum(sess.run(model.W2_masked).reshape(-1) > 0))
                print('    W3_masked features', sum(sess.run(model.W3_masked).reshape(-1) > 0))
              print("h1", np.std(sess.run(model.h1, feed_dict=nat_dict)))
              #print('    W1 features', sum(sess.run(model.W1).reshape(-1) > 1e-12))
              #print('    W2 features', sum(sess.run(model.W2).reshape(-1) > 1e-12))
              #print('    W3 features', sum(sess.run(model.W3).reshape(-1) > 1e-12))
              #Validation
              if val_acc > best_val_acc:
                print("New best val acc is", val_acc)
                best_val_acc = val_acc
                num_iters = ii
                test_acc = sess.run(model.accuracy, feed_dict=test_dict)
                print("New best test acc is", test_acc)
                logits_acc[experiment] = sess.run(model.logits, feed_dict=test_dict)
                W1_acc[experiment] = sess.run(model.W1).reshape(-1)
                W2_acc[experiment] = sess.run(model.W2).reshape(-1)
                W3_acc[experiment] = sess.run(model.W3).reshape(-1)
                preds[experiment] = sess.run(model.y_pred, feed_dict=test_dict)
    
              #Tensorboard Summaries
              #summary = tf.Summary(value=[
              #    tf.Summary.Value(tag='Xent', simple_value= nat_xent),])
              #summary1 = tf.Summary(value=[
              #    tf.Summary.Value(tag='Xent', simple_value= max_xent),])
              #summary2 = tf.Summary(value=[
              #    tf.Summary.Value(tag='Accuracy', simple_value= nat_acc*100)])
              #summary3 = tf.Summary(value=[
              #    tf.Summary.Value(tag='Accuracy', simple_value= test_acc*100)])
              #summary_writer.add_summary(summary, global_step.eval(sess))
              #summary_writer1.add_summary(summary1, global_step.eval(sess))
              #summary_writer2.add_summary(summary2, global_step.eval(sess))
              #summary_writer3.add_summary(summary3, global_step.eval(sess))
    
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
          theta= sess.run(model.theta, feed_dict=test_dict)
          test_accs[experiment] = test_acc  * 100
          thetas[experiment] = theta
          iterations[experiment] = num_iters
          avg_test_acc += test_acc
  elif model_type == "cnn":
      logits_acc = np.zeros((config['num_experiments'], 10000, 10))
      conv11_acc = np.zeros((config['num_experiments'], 3*3*num_channels*args.cnn_size))
      conv12_acc = np.zeros((config['num_experiments'], 3*3*args.cnn_size*args.cnn_size))
      conv21_acc = np.zeros((config['num_experiments'], 3*3*args.cnn_size*args.cnn_size))
      conv22_acc = np.zeros((config['num_experiments'], 3*3*args.cnn_size*args.cnn_size))
      conv31_acc = np.zeros((config['num_experiments'], 3*3*args.cnn_size*args.cnn_size))
      conv32_acc = np.zeros((config['num_experiments'], 3*3*args.cnn_size*args.cnn_size))
      fc1_acc = np.zeros((config['num_experiments'], int((int(int((pixels_x+1)/2)+1)/2+1)/2) * int((int(int((pixels_y+1)/2)+1)/2+1)/2) * args.cnn_size * args.fc_size))
      fc2_acc = np.zeros((config['num_experiments'], args.fc_size * 10))

      #TODO: replace 10000 by the actual size of test set -- solved?
      preds = np.zeros((config['num_experiments'], data.test.images.shape[0]))
    
      for experiment in range(num_experiments):
        print("Experiment", experiment)
    
        # Setting up the Tensorboard and checkpoint outputs
        model_dir = config['model_dir'] + str(datetime.now().date())
        if not os.path.exists(model_dir):
          os.makedirs(model_dir)
    
        with tf.Session() as sess:
          # Initialize the summary writer, global variables, and our time counter.
          #summary_writer = tf.summary.FileWriter(model_dir + "/Xent")
          #summary_writer1 = tf.summary.FileWriter(model_dir+ "/Max_Xent")
          #summary_writer2 = tf.summary.FileWriter(model_dir+ "/Accuracy")
          #summary_writer3 = tf.summary.FileWriter(model_dir+ "/Test_Accuracy")
          sess.run(tf.global_variables_initializer())
          training_time = 0.0
    
          # Main training loop
          best_val_acc = 0
          test_acc = 0
          num_iters = 0
          for ii in range(max_num_training_steps):
            x_batch, y_batch = data.train.next_batch(batch_size)
            y_batch = y_batch.reshape(-1) 
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
    
            # Output
            if ii % num_output_steps == 0:
              nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
              val_acc = sess.run(model.accuracy, feed_dict=val_dict)
              nat_xent = sess.run(model.xent, feed_dict=nat_dict)
              dual_xent = sess.run(model.dual_xent, feed_dict=nat_dict)
              MC_xent = sess.run(model.MC_xent, feed_dict=nat_dict)
              print('Step {}:    ({})'.format(ii, datetime.now()))
              print('    training nat accuracy {:.4}'.format(nat_acc * 100))
              print('    validation nat accuracy {:.4}'.format(val_acc * 100))
              print('    Nat Xent {:.4}'.format(nat_xent))
              print('    Max Xent upper bound with dual {:.4}'.format(dual_xent))
              print('    Max Xent lower bound with Monte Carlo {:.4}'.format(MC_xent))
    
              #Validation
              if val_acc > best_val_acc:
                print("New best val acc is", val_acc)
                best_val_acc = val_acc
                num_iters = ii
                test_acc = sess.run(model.accuracy, feed_dict=test_dict)
                print("New best test acc is", test_acc)
                logits_acc[experiment] = sess.run(model.logits, feed_dict=test_dict)
                conv11_acc[experiment] = sess.run(model.conv11.kernel).reshape(-1)
                conv12_acc[experiment] = sess.run(model.conv12.kernel).reshape(-1)
                conv21_acc[experiment] = sess.run(model.conv21.kernel).reshape(-1)
                conv22_acc[experiment] = sess.run(model.conv22.kernel).reshape(-1)
                conv31_acc[experiment] = sess.run(model.conv31.kernel).reshape(-1)
                conv32_acc[experiment] = sess.run(model.conv32.kernel).reshape(-1)
                fc1_acc[experiment] = sess.run(model.fc1.kernel).reshape(-1)
                fc2_acc[experiment] = sess.run(model.fc2.kernel).reshape(-1)
                preds[experiment] = sess.run(model.y_pred, feed_dict=test_dict)
    
              #Tensorboard Summaries
              #summary = tf.Summary(value=[
              #    tf.Summary.Value(tag='Xent', simple_value= nat_xent),])
              #summary1 = tf.Summary(value=[
              #    tf.Summary.Value(tag='Xent', simple_value= max_xent),])
              #summary2 = tf.Summary(value=[
              #    tf.Summary.Value(tag='Accuracy', simple_value= nat_acc*100)])
              #summary3 = tf.Summary(value=[
              #    tf.Summary.Value(tag='Accuracy', simple_value= test_acc*100)])
              #summary_writer.add_summary(summary, global_step.eval(sess))
              #summary_writer1.add_summary(summary1, global_step.eval(sess))
              #summary_writer2.add_summary(summary2, global_step.eval(sess))
              #summary_writer3.add_summary(summary3, global_step.eval(sess))
    
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
          theta= sess.run(model.theta, feed_dict=test_dict)
          test_accs[experiment] = test_acc  * 100
          thetas[experiment] = theta
          iterations[experiment] = num_iters
          avg_test_acc += test_acc      


  avg_test_acc  = avg_test_acc/num_experiments
  print('  Average testing accuracy {:.4}'.format(avg_test_acc  * 100))
  print('  Theta values', thetas)
  #print('  individual accuracies: \n', test_accs)
  std = np.array([float(test_accs[k]) for k in test_accs]).std()
  print('Test Accuracy std {:.2}'.format(np.array([float(test_accs[k]) for k in test_accs]).std()))
  print("Logits std", np.mean(np.mean(np.std(logits_acc, axis=0), axis=0)))
  logit_stability =  np.mean(np.std(logits_acc, axis=0), axis=0)
  gini_stability = total_gini(preds.transpose())
  if model_type == "ff":
      w1_stability = np.mean(np.std(W1_acc, axis=0), axis=0)
      w2_stability = np.mean(np.std(W2_acc, axis=0), axis=0)
      w3_stability = np.mean(np.std(W3_acc, axis=0), axis=0)
      print("W1 std", w1_stability)
      print("W2 std", w2_stability)
      print("W3 std", w3_stability)
      print("Gini stability", gini_stability)
      file = open(str('results' + data_set + '.csv'), 'a+', newline ='')
      with file:
        writer = csv.writer(file)
        writer.writerow([stable, num_experiments, args.train_size, batch_size, subset_ratio, avg_test_acc, test_accs, std, thetas, max_num_training_steps, iterations, w1_stability, w2_stability, w3_stability, logit_stability, gini_stability, ])

  elif model_type == "cnn":
      conv11_stability = np.mean(np.std(conv11_acc, axis=0), axis=0)
      conv12_stability = np.mean(np.std(conv12_acc, axis=0), axis=0)
      conv21_stability = np.mean(np.std(conv21_acc, axis=0), axis=0)
      conv22_stability = np.mean(np.std(conv22_acc, axis=0), axis=0)
      conv31_stability = np.mean(np.std(conv31_acc, axis=0), axis=0)
      conv32_stability = np.mean(np.std(conv31_acc, axis=0), axis=0)
      fc1_stability = np.mean(np.std(fc1_acc, axis=0), axis=0)
      fc2_stability = np.mean(np.std(fc2_acc, axis=0), axis=0)

      print("conv11 std", conv11_stability)
      print("conv12 std", conv12_stability)
      print("conv21 std", conv21_stability)
      print("conv22 std", conv22_stability)
      print("conv31 std", conv31_stability)
      print("conv32 std", conv32_stability)
      print("fc1 std", fc1_stability)
      print("fc2 std", fc2_stability)
      print("Gini stability", gini_stability)      

      file = open(str('results_cnn' + data_set + '.csv'), 'a+', newline ='')
      with file:
        writer = csv.writer(file)
        writer.writerow([stable, num_experiments, args.train_size, batch_size, subset_ratio, avg_test_acc, test_accs, std, thetas, max_num_training_steps, iterations, conv11_stability, conv12_stability, conv21_stability, conv22_stability,conv31_stability,conv32_stability, fc1_stability, fc2_stability, logit_stability, gini_stability, ])
