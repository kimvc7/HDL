"""Trains a model with cross validation, saving checkpoints and tensorboard summaries along the way."""


# Import packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../utils')

import importlib.machinery
import importlib.util
from datetime import datetime
import json
import os
from timeit import default_timer as timer
import numpy as np
import input_data
import itertools
from utils_init import *
import utils_model
import utils_print

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load config file with static parameters
with open('config.json') as config_file:
	config = json.load(config_file)

#Import Network Model
network_path = config["network_path"]
#loader = importlib.machinery.SourceFileLoader('Model', './Networks/' + network_path + '.py')
spec = importlib.util.spec_from_file_location(network_path, './Networks/' + network_path + '.py')
network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_module)


# Parse arguments
parser = define_parser()
args = parser.parse_args()


# Set up training parameters
seed, max_train_steps, num_output_steps, num_summary_steps, num_check_steps = read_config_train(config)
rho, is_stable, learning_rate, l0, l2, batch_range, stab_ratio_range, dropout, network_size = read_train_args(args)
data_set, train_size, val_size = read_data_args(args)


# Training Initializitation
global_step = tf.Variable(1, name="global_step")
min_train_steps = int(0.4*max_train_steps)
tf.set_random_seed(seed)

if rho == 0 and not is_stable and l0 == 0:
	min_train_steps = 0


# Training Loop for cross validation
for batch_size, subset_ratio in itertools.product(batch_range, stab_ratio_range): 
	
	print("Batch Size:", batch_size, " ; stability subset ratio:", subset_ratio, " ; dropout value:", dropout)

	# Set up data 
	data = input_data.load_data_set(training_size = train_size, validation_size= val_size, data_set=data_set, seed=seed)
	num_features = data.train.images.shape[1]
	num_classes = np.unique(data.train.labels).shape[0]

	# Set up model and optimizer
	model = network_module.Model(num_classes, batch_size, network_size, subset_ratio, num_features, dropout, l2, l0, rho)
	loss = utils_model.get_loss(model, args)
	network_vars, sparse_vars, stable_var = read_config_network(config, model)

	var_list = network_vars
	if args.l0 > 0:
		var_list = var_list + sparse_vars
	if args.is_stable:
		var_list = var_list + stable_var

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list) 


	# Set up experiments
	total_test_acc = 0
	num_experiments, dict_exp, output_dir = init_experiements(config, args, num_classes, num_features, data) 

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	# Training loop for each fold
	for experiment in range(num_experiments):

		# Set up summary and results folder
		directory = output_dir + '/exp_' + str(experiment) + '_l2reg_' + str(l2)
		summary_writer = tf.summary.FileWriter(directory)
		saver = tf.train.Saver(max_to_keep=3)

		# Shuffle and split training/vaidation/testing sets

		seed_i = seed*(experiment+1)
		data = input_data.load_data_set(training_size = train_size, validation_size=val_size, data_set=data_set, seed=seed_i)

		# Set up data sets for validation and testing
		val_dict = {model.x_input: data.validation.images,
					  model.y_input: data.validation.labels.reshape(-1)}

		test_dict = {model.x_input: data.test.images,
					  model.y_input: data.test.labels.reshape(-1)}


		# Initialize tensorflow session
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			training_time = 0.0
		
			best_val_acc, test_acc, num_iters = 0, 0, 0


			# Iterate through each data batch
			for train_step in range(max_train_steps):

				x_batch, y_batch = data.train.next_batch(batch_size)
				nat_dict = {model.x_input: x_batch,
							model.y_input: y_batch}

				
				if train_step % num_output_steps == 0:

					# Update results
					dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment)

					# Print and Save current status
					utils_print.print_metrics(sess, model, nat_dict, val_dict, test_dict, train_step, args, summary_writer, dict_exp, experiment, global_step)
					saver.save(sess, directory+ '/checkpoints/checkpoint', global_step=global_step)
				  
					# Track best validation accuracy
					val_acc = sess.run(model.accuracy, feed_dict=val_dict)
					if val_acc > best_val_acc and train_step > min_train_steps:
						best_val_acc = val_acc
						num_iters = train_step
						test_acc = sess.run(model.accuracy, feed_dict=test_dict)

					# Check time
					if train_step != 0:
						print('    {} examples per second'.format(num_output_steps * batch_size / training_time))
						training_time = 0.0

				# Train model with current batch
				start = timer()
				sess.run(optimizer, feed_dict=nat_dict)
				end = timer()
				training_time += end - start
		
		
			# Output final results for current experiment
			utils_print.update_dict_output(dict_exp, experiment, sess, test_acc, model, test_dict, num_iters)
			total_test_acc += test_acc
			x_test, y_test = data.test.images, data.test.labels.reshape(-1)
			best_model = utils_model.get_best_model(dict_exp, experiment, args, num_classes, batch_size, subset_ratio, num_features, spec, network_module)
			utils_print.update_adv_acc(args, best_model, x_test, y_test, experiment, dict_exp)


	utils_print.print_stability_measures(dict_exp, args, num_experiments, batch_size, subset_ratio, total_test_acc, max_train_steps, network_path)
