"""Trains a model with cross validation, saving checkpoints and tensorboard summaries along the way."""


# Import packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('../utils')

import importlib.util
import json
import os
from timeit import default_timer as timer
import numpy as np
import input_data
import itertools
import pickle
from utils_init import *
import utils_model
from utils_nn_model import *
import utils_print

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load config file with static parameters
with open('config.json') as config_file:
	config = json.load(config_file)



# Parse arguments
parser = define_parser()
args = parser.parse_args()


# Set up training parameters
seed, num_epochs, num_output_steps, num_summary_steps, num_check_steps, final_temp, num_rounds, rewind_epoch = read_config_train(config)
args, rho, is_stable, learning_rate, l0, l2, batch_range, stab_ratio_range, dropout, network_size, pool_size, network_path = read_train_args_hypertuning(args)
data_set, train_size, val_size = read_data_args(args)
data_shape_size = 4

if l0 == 0:
	num_rounds = 1


#Import Network Model
spec = importlib.util.spec_from_file_location(network_path, './Networks/' + network_path + '.py')
network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_module)


# Training Initializitation
global_step = tf.Variable(1, name="global_step")
tf.set_random_seed(seed)



# Training Loop for cross validation
for batch_size, subset_ratio in itertools.product(batch_range, stab_ratio_range): 
	
	print("Batch Size:", batch_size, " ; stability subset ratio:", subset_ratio, " ; dropout value:", dropout)

	# Set up data 
	data, data_shape = input_data.load_data_set(training_size = train_size, validation_size= val_size, data_set=data_set, seed=seed)

	if len(data_shape) != data_shape_size:
		data_shape = None
	else:
		data_shape = data_shape[1:]

	num_features = data.train.images.shape[1]
	num_classes = np.unique(data.train.labels).shape[0]

	# Find number of training steps and temeprature increase factor.
	iters_per_epoch =  int(data.train.images.shape[0]/batch_size)
	max_train_steps = int(iters_per_epoch * num_epochs)
	temp_increase = final_temp**(1./(num_epochs-1))
	print("Training size is: ", data.train.images.shape[0])
	print("Number of epochs is: ", num_epochs)
	print("Number of iterations per round is: ", max_train_steps)
 

	# Set up experiments
	num_experiments, dict_exp, output_dir = init_experiments(config, args, num_classes, num_features, data)

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
		data, xx = input_data.load_data_set(training_size = train_size, validation_size=val_size, data_set=data_set, seed=seed_i)

		#Inizialize weights storesd at end of each round and weights to rewind in final round.
		stored_weights = None
		rewind_weights = None

		for round_ in range(num_rounds):
			print("__________________________________________________________")
			print("__________________________________________________________")
			print("Starting round #", round_, " for experiment #", experiment)

			ticket = False
			temp = 1
			print("---------------------------------")
			print("Temperature initialized to ", temp)

			if (round_ == num_rounds - 1) and (num_rounds >1):
				ticket = True
				stored_weights['network_weights'] = rewind_weights['network_weights']
				stored_weights['network_biases'] = rewind_weights['network_biases']
				stored_weights['stability_variable'] = rewind_weights['stability_variable']


			# Set up model 
			model = network_module.Model(num_classes, batch_size, network_size, pool_size, subset_ratio, num_features, dropout, l2, l0, rho, data_shape, ticket, stored_weights)
			loss = utils_model.get_loss(model, args)
			network_vars, sparse_vars, stable_var = read_config_network(config, args, model)

			# Set up data sets for validation and testing
			val_dict = {model.x_input: data.validation.images,
					  	model.y_input: data.validation.labels.reshape(-1),
					  	model.temp: 1}

			test_dict = {model.x_input: data.test.images,
					  	model.y_input: data.test.labels.reshape(-1),
					  	model.temp: 1}


			#Set up optimizer
			var_list = network_vars
			if (args.l0 > 0) and (round_ != num_rounds - 1):
				var_list = var_list + sparse_vars
			if args.is_stable:
				var_list = var_list + stable_var

			optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=var_list)


			# Initialize tensorflow session
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				training_time = 0.0
			
				best_val_acc = 0


				# Iterate through each data batch
				for train_step in range(max_train_steps):


					if (train_step % iters_per_epoch == 0) and (train_step != 0):
						temp = temp_increase * temp
						print("-----------------------------------")
						print("Temperature has increased to ", temp)


					x_batch, y_batch = data.train.next_batch(batch_size)
					nat_dict = {model.x_input: x_batch,
								model.y_input: y_batch,
								model.temp: temp}

					if (round_ == 0) and (train_step == int(iters_per_epoch * rewind_epoch)):
						rewind_weights = store_network(model, args, sess, nat_dict, True)
					
					if  (train_step % num_output_steps == 0): 


						# Print and Save current status
						utils_print.print_metrics(sess, model, nat_dict, val_dict, test_dict, train_step, args, summary_writer, experiment, global_step)
						#saver.save(sess, directory+ '/checkpoints/checkpoint', global_step=global_step)
					  
						# Track best validation accuracy
						val_acc = sess.run(model.accuracy, feed_dict=val_dict)
						if l0 >0:
							print("W1mask sum is: ", np.sum(sess.run(model.W1_masked, feed_dict=val_dict)))
						print("theta: ", np.sum(sess.run(model.theta, feed_dict=val_dict)))
						print("W1 sum is: ", np.sum(sess.run(model.W1, feed_dict=val_dict)))

						if val_acc > best_val_acc and (round_ == num_rounds-1):
							print("best")
							best_val_acc = val_acc
							# Update best results
							dict_exp = utils_model.update_dict(dict_exp, args, sess, model, test_dict, experiment, train_step)

						# Check time
						if train_step != 0:
							print('    {} examples per second'.format(num_output_steps * batch_size / training_time))
							training_time = 0.0

					# Train model with current batch
					start = timer()
					sess.run(optimizer, feed_dict=nat_dict)
					end = timer()
					training_time += end - start

					if (round_ != num_rounds - 1) and (train_step == max_train_steps -1):
						prune = (round_ != num_rounds - 2)
						# see utils_models for this function
						stored_weights = store_network(model, args, sess, nat_dict, prune)

		
		
		# Output final results for current experiment
		x_test, y_test = data.test.images, data.test.labels.reshape(-1)
		best_model = utils_model.get_best_model(dict_exp, experiment, args, num_classes, batch_size, subset_ratio, num_features, spec, network_module, network_size, pool_size, data_shape)
		dict_exp = utils_print.update_best_acc(args, best_model, x_test, y_test, experiment, dict_exp)

		# Save weights to a pickle file
		with open('saved_weights.pkl', 'wb') as f:
			pickle.dump(stored_weights, f)

	utils_print.print_stability_measures(dict_exp, args, num_experiments, batch_size, subset_ratio, max_train_steps, network_path)
