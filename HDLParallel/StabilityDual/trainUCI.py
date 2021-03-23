"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
#import shutil
from timeit import default_timer as timer

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import input_data
import itertools
import utils_model
import utils_print

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--gnum", type=int, default=1, help='which gen param of 72 [0,71]')
parser.add_argument("--mnum", type=int, default=1, help='which method/data of 34 [0,33]')

parser.add_argument("--batch_range", type=int, nargs='+', default=[64],
                            help="batch range")

parser.add_argument("--ratio_range", type=float, nargs='+', default=[0.8],
                            help="ratio range")

parser.add_argument("--model", "-m", type=str, required=True, choices=["ff", "cnn"],
                            help="model type, either ff or cnn")

parser.add_argument("--stable", action="store_true",
                            help="stable version")

parser.add_argument("--dropout", type=float, default=1,
                            help="dropout rate, 1 is no dropout, 0 is all set to 0")

parser.add_argument("--robust", "-r", type=float, default=0,
                            help="Uncertainty set parameter for training robustness.")

parser.add_argument("--robust_test", "-rtest", type=float,  nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                            help="Uncertainty set parameter for evaluating robustness.")

parser.add_argument("--l2", type=float, default=0,
                            help="l2 regularization rate")

parser.add_argument("--l0", type=float, default=0,
                            help="l0 regularization rate")

parser.add_argument("--reg_stability", type=float, default=0,
                            help="reg stability regularization rate")

parser.add_argument("--num_subsets", type=int, default=1,
                            help="number of subsets for Monte Carlo")

parser.add_argument("--l1_size", type=int, default=256,
                            help="number of nodes in the first layer, 784 -> l1_size")

parser.add_argument("--l2_size", type=int, default=128,
                            help="number of nodes in the first layer, l1_size -> l2_size")

parser.add_argument("--cnn_size", type=int, default=32,
                            help="number of filters in the cnn layers for cnn")

parser.add_argument("--fc_size", type=int, default=128,
                            help="number of nodes in the dense layer for cnn")

parser.add_argument("--data_set", type=str, default="mnist",
                            help="number of subsets")

parser.add_argument("--MC", action="store_true",
                            help="Monte Carlo version")

parser.add_argument("--train_size", type=float, default=0.80,
                            help="training percent of the data")

parser.add_argument("--lr", type=float, default=0.001,
                            help="Adam lr")

parser.add_argument("--val_size", type=float, default=0.20,
                            help="validation percent of the data e.g., 0.25 means 0.25*traning size")
args = parser.parse_args()
with open('config.json') as config_file:
    config = json.load(config_file)
    
    
gen_param = []
for batchsize in [64,128,16]:
    for l_r in [1e-3,1e-4]:
        for l_2 in [0,1e-5,1e-4,1e-3]:
            for drop_out in [1,]:
                for nnsize in [(64,32),(256,128),(128,64)]:
                    gen_param.append((batchsize,l_r,l_2,drop_out,nnsize))
                        
ratiorange = 0.8
gen_param = gen_param[args.gnum]

robust,stable,l0 = -1,-1,-1
if args.mnum==0: robust,stable,l0 = 0,0,0

elif args.mnum==1: robust,stable,l0 = 0,1,0

elif args.mnum==2: robust,stable,l0 = 0,0,1e-4
elif args.mnum==3: robust,stable,l0 = 0,0,1e-5
elif args.mnum==4: robust,stable,l0 = 0,0,1e-6

elif args.mnum>=5 and args.mnum<=9:
    robust=10**(-1*(10-args.mnum))
    stable,l0=0,0

elif args.mnum>=10 and args.mnum<=33:
    rob_range,l0_range = [0,1e-5,1e-4,1e-3,1e-2,1e-1],[0,1e-4,1e-5,1e-6]
    combos = [(i,j) for i in rob_range for j in l0_range][1:]
    
    robust=combos[args.mnum-10][0]
    l0=combos[args.mnum-10][1]
    stable=1
else:
    print('invalid mnum input')
    1/0
    
    
    


print(args)
if args.model == "ff":
    from NN_model import Model
    #from L2NN_model import Model
elif args.model == "cnn":
    from CNN_model import Model
    assert(tf.keras.backend.image_data_format() == "channels_last")




args.batch_range = [gen_param[0]]
args.lr = gen_param[1]
args.l2 = gen_param[2]
args.dropout = gen_param[3]
args.l1_size = gen_param[4][0]
args.l2_size = gen_param[4][1]

args.robust = robust
args.stable = stable
args.l0 = l0

args.ratio_range = [ratiorange]
args.data_set=args.data_set
args.MC=False




# Setting up training parameters
seed = config['random_seed']
tf.set_random_seed(seed)
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
testing_size = config['testing_size']
data_set = args.data_set
num_subsets = args.num_subsets
#initial_learning_rate = config['initial_learning_rate']
eta = config['constant_learning_rate']
learning_rate = tf.train.exponential_decay(args.lr,
 0, 5, 0.85, staircase=True)
theta = args.stable and (not args.MC)
num_channels, pixels_x, pixels_y = 0, 0, 0
reshape = True
global_step = tf.Variable(1, name="global_step")
min_num_training_steps = int(0.4*max_num_training_steps)
if args.robust == 0 and not args.stable and args.l0 == 0:
  min_num_training_steps = 0


for batch_size, subset_ratio in itertools.product(args.batch_range, args.ratio_range): #Parameters chosen with validation
  print(batch_size, subset_ratio, args.dropout)


  #Setting up the data and the model
  if args.model == "ff":
      data = input_data.load_data_set(training_size = args.train_size, validation_size=args.val_size, data_set=data_set, reshape=reshape, seed=seed)
      num_features = data.train.images.shape[1]
      num_classes = np.unique(data.train.labels).shape[0]
      model = Model(num_classes, num_subsets, batch_size, args.l1_size, args.l2_size, subset_ratio, num_features, args.dropout, args.l2, args.l0, args.robust, args.reg_stability)
      var_list = [model.W1, model.b1, model.W2, model.b2, model.W3, model.b3]
      if args.l0 > 0:
          var_list = [model.W1, model.log_a_W1, model.b1, model.log_a_W2, model.W2, model.b2, model.W3, model.log_a_W3, model.b3]
    
  elif args.model == "cnn":
      reshape = False
      data = input_data.load_data_set(training_size = args.train_size, validation_size=args.val_size, data_set=data_set, reshape=reshape, seed=seed)
      print(data.train.images.shape)
      num_classes = np.unique(data.train.labels).shape[0]
      num_features = data.train.images.shape[1]
      pixels_x = data.train.images.shape[1]
      pixels_y = data.train.images.shape[2]
      num_channels = data.train.images.shape[3]

      model = Model(num_subsets, batch_size, args.cnn_size, args.fc_size, subset_ratio, pixels_x, pixels_y, num_channels, args.dropout, args.l2, theta, args.robust)


  #Returns the right loss depending on MC or dual or nothing
  loss = utils_model.get_loss(model, args)


  # Setting up the optimizer
  if (args.stable and not args.MC) or (args.model == "cnn"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss + model.regularizer, global_step=global_step)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss + model.regularizer, global_step=global_step, var_list=var_list)

  
  #Initializing loop variables.
  avg_test_acc = 0
  num_experiments = config['num_experiments']
  dict_exp = utils_model.create_dict(args, num_classes, data.train.images.shape, data.test.images.shape)
  output_dir = 'outputs/logs/' + str(args.data_set) + '/' + str(datetime.now())
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)


  for experiment in range(num_experiments):
    acc = 0
    seed_i = seed*(experiment+1)
    data = input_data.load_data_set(training_size = args.train_size, validation_size=args.val_size, data_set=data_set, reshape=reshape, seed=seed_i)

    #Setting up data for testing and validation
    val_dict = {model.x_input: data.validation.images,
                  model.y_input: data.validation.labels.reshape(-1)}
    test_dict = {model.x_input: data.test.images[:testing_size],
                  model.y_input: data.test.labels[:testing_size].reshape(-1)}

    directory = output_dir + '/exp_' + str(experiment) + '_l2reg_' + str(args.l2)
    summary_writer = tf.summary.FileWriter(directory)
    saver = tf.train.Saver(max_to_keep=3)

    with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          training_time = 0.0
    
          # Main training loop
          best_val_acc, test_acc, num_iters = 0, 0, 0

          for ii in range(max_num_training_steps):
            x_batch, y_batch = data.train.next_batch(batch_size)
            if args.model == "cnn":
                y_batch = y_batch.reshape(-1)
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}

    
            # Output
            if ii % num_output_steps == 0:

              val_acc = utils_print.print_metrics(sess, model, nat_dict, val_dict, test_dict, ii, args, summary_writer, global_step)
              saver.save(sess, directory+ '/checkpoints/checkpoint', global_step=global_step)
              
               #Validation
              if val_acc >= best_val_acc and ii > min_num_training_steps:
                print("New best val acc is", val_acc)
                best_val_acc = val_acc
                num_iters = ii
                test_acc = sess.run(model.accuracy, feed_dict=test_dict)

                print("New best test acc is", test_acc)
                dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment)
                if val_acc >= 99.5:
                    acc+=1

              if ii != 0:
                print('    {} examples per second'.format(
                    num_output_steps * batch_size / training_time))
                training_time = 0.0

            # Actual training step
            start = timer()
            sess.run(optimizer, feed_dict=nat_dict)
            end = timer()
            training_time += end - start
            if acc >= 5:
                break
    
          #Output test results
          utils_print.update_dict_output(dict_exp, experiment, sess, test_acc, model, test_dict, num_iters)
          avg_test_acc += test_acc
          x_test, y_test = data.test.images[:testing_size], data.test.labels[:testing_size].reshape(-1)
          best_model = utils_model.get_best_model(dict_exp, experiment, args, num_classes, num_subsets, batch_size, subset_ratio, 
            num_features, theta, num_channels, pixels_x, pixels_y)
          utils_print.update_adv_acc(args, best_model, x_test, y_test, experiment, dict_exp)

  utils_print.print_stability_measures(dict_exp, args, num_experiments, batch_size, subset_ratio, avg_test_acc, max_num_training_steps)
