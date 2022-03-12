import argparse
import utils_model
from datetime import datetime


VGG16 = ((3, 3, 64), (3, 3, 64), (3, 3, 128), (3, 3, 128), (3, 3, 256), (3, 3, 256), #(3, 3, 256),
		(3, 3, 512), (3, 3, 512), (3, 3, 512), (3, 3, 512), (3, 3, 512), (3, 3, 512), (4096,), (4096,), (1000,))
VGG16_POOL = (False, True, False, True, False, True, False, False, True, False, False, True, False, False, False)

#ALexNet architecture
ALEX = ((5, 5, 32), (5, 5, 64), (1024,))
ALEX_POOL = (True, True, False)

#VGG3 architecture
VGG3 = ((3, 3, 64), (3, 3, 64), (1000,))
VGG3_POOL = (True, True, False)

#VGG7 architecture
VGG7 = ((3, 3, 32), (3, 3, 32), (3, 3, 64), (3, 3, 64), (3, 3, 128), (3, 3, 128), (128,))
VGG7_POOL = (False, True, False, True, False, True, False)

MLP = (256, 128)
MLP_POOL = (False, False)


NN = {"MLP":MLP, "ALEX":ALEX, "VGG3":VGG3, "VGG7":VGG7, "VGG16":VGG16}
NN_POOL = {"MLP":MLP_POOL, "ALEX":ALEX_POOL, "VGG3":VGG3_POOL, "VGG7":VGG7_POOL, "VGG16":VGG16_POOL}
NN_PATH = {"MLP":"MLP_model", "ALEX":"CNN_model", "VGG3":"CNN_model", "VGG7":"CNN_model", "VGG16":"CNN_model"}


def define_parser():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("--batch_range", type=int, nargs='+', default=[64], help="batch range")

	parser.add_argument("--network_type", type=str,  default="VGG3", help="network used for training (ALEX, VGG3, VGG16, MLP)")

	parser.add_argument("--stab_ratio_range", type=float, nargs='+', default=[0.8], help="stability ratio range")

	parser.add_argument("--is_stable", action="store_true", help="stable version")

	parser.add_argument("--dropout", type=float, default=1, help="dropout rate, 1 is no dropout, 0 is all set to 0")

	parser.add_argument("--rho", type=float, default=0, help="Radius of the uncertainty set for robust training.")

	parser.add_argument("--robust_test", "-rtest", type=float,  nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1], help="radius of the uncertainty set for robust testing.")

	parser.add_argument("--l2", type=float, default=0, help="l2 regularization parameter")

	parser.add_argument("--l0", type=float, default=0, help="l0 regularization parameter")

	parser.add_argument("--reg_stability", type=float, default=0, help="reg stability regularization parameter")

	parser.add_argument("--data_set", type=str, default="mnist", help="dataset name")

	parser.add_argument("--train_size", type=float, default=1, help="percentage of data used of training")

	parser.add_argument("--lr", type=float, default=0.0001, help="learning Rate used for the optimizer")

	parser.add_argument("--val_size", type=float, default=(1/6), help="percentage of data used for validation")

	parser.add_argument("--exp_id", type=int, default=0, help="experiment id corresponding to a predefined config of parameter, decided for the paper's experiments")

	return parser


def read_config_train(config):
	seed = config['random_seed']
	max_train_steps = config['max_num_training_steps']
	num_output_steps = config['num_output_steps']
	num_summ_steps = config['num_summary_steps']
	num_check_steps = config['num_checkpoint_steps']

	return seed, max_train_steps, num_output_steps, num_summ_steps, num_check_steps


def init_vars(n):
	network_vars_w = ["W"+str(i+1) for i in range(n)]
	network_vars_b = ["b"+str(i+1) for i in range(n)]
	stable_var = "theta"
	sparse_vars = ["log_a_W"+str(i+1) for i in range(n)]

	return network_vars_w, network_vars_b, stable_var, sparse_vars


def read_config_network(config, args, model):
	network_size = list(NN[args.network_type])
	name_vars_w, name_vars_b, name_stable_var, name_sparse_vars = init_vars(len(network_size)+1)
	network_vars_w = [getattr(model, var) for var in  name_vars_w]
	network_vars_b = [getattr(model, var) for var in  name_vars_b]
	network_vars = network_vars_w + network_vars_b 
	sparsity_vars = [getattr(model, var) for var in  name_sparse_vars]
	stable_var = [getattr(model, name_stable_var)]

	return network_vars, sparsity_vars, stable_var

def read_train_args(args):
	rho = args.rho
	is_stable = args.is_stable
	learning_rate = args.lr
	l0 = args.l0
	l2 = args.l2
	batch_range = args.batch_range
	stab_ratio_range = args.stab_ratio_range
	dropout = args.dropout
	network_size = list(NN[args.network_type])
	pool_size = list(NN_POOL[args.network_type])
	model_path = NN_PATH[args.network_type]

	return rho, is_stable, learning_rate, l0, l2, batch_range, stab_ratio_range, dropout, network_size, pool_size, model_path

def read_data_args(args):
	data_set = args.data_set
	train_size = args.train_size
	val_size = args.val_size

	return data_set, train_size, val_size

def init_experiments(config, args, num_classes, num_features, data):
	num_experiments = config['num_experiments']
	dict_exp = utils_model.create_dict(args, num_classes, num_features, data.train.images.shape, data.test.images.shape)
	output_dir = 'outputs/logs/' + str(args.data_set) + '/' + str(datetime.now())
	return num_experiments, dict_exp, output_dir

def read_train_args_hypertuning(args):
	param_combos = produce_configs()
	if args.exp_id >= 0:
		gen_param = param_combos[args.exp_id]
		args.batch_range = [gen_param[0]]
		args.lr = gen_param[1]
		args.l2 = gen_param[2]
		args.dropout = gen_param[3]
		args.is_stable = gen_param[4]
		args.l0 = gen_param[5]
		#args.r = gen_param[6]
		args.rho = gen_param[6]

	rho = args.rho
	is_stable = args.is_stable
	learning_rate = args.lr
	l0 = args.l0
	l2 = args.l2
	batch_range = args.batch_range
	stab_ratio_range = args.stab_ratio_range
	dropout = args.dropout
	network_size = list(NN[args.network_type])
	pool_size = list(NN_POOL[args.network_type])
	model_path = NN_PATH[args.network_type]

	print(args)
	return args, rho, is_stable, learning_rate, l0, l2, batch_range, stab_ratio_range, dropout, network_size, pool_size, model_path

def produce_configs():
	gen_param = []
	for batchsize in [64,256]:
		for lr in [1e-4, 1e-3, 1e-2]:
			for l0 in [0]:#, 1e-6, 1e-5, 1e-4]:
				for l2 in [0, 1e-5, 1e-4, 1e-3]:
					for drop_out in [1]:
						for stable in [0,1]:
							for r in [0]:#, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
								gen_param.append((batchsize, lr, l2, drop_out, stable, l0, r))
	return gen_param


