import numpy as np
import json

with open('config.json') as config_file:
    config = json.load(config_file)

def get_loss(model, args):
    loss = model.xent
    if args.robust and not args.stable:
        loss = model.robust_xent
    if args.MC:
        assert(not args.robust)
        loss = model.MC_xent
    elif args.stable:
        if args.robust:
            loss = model.robust_stable_xent
        else:
            loss = model.dual_xent

    return loss


def create_dict(args, train_shape, test_size):
    dict_exp = {}
    dict_exp['logits_acc'] = np.zeros((config['num_experiments'], test_size[0], 10))
    dict_exp['W1_acc'] = np.zeros((config['num_experiments'], train_shape[1] * args.l1_size))
    dict_exp['W2_acc'] = np.zeros((config['num_experiments'], args.l1_size * args.l2_size))
    dict_exp['W3_acc'] = np.zeros((config['num_experiments'], args.l2_size * 10))
    dict_exp['preds'] = np.zeros((config['num_experiments'], test_size[0]))
    dict_exp['test_accs'] = np.zeros(config['num_experiments'])
    dict_exp['adv_test_accs'] = np.zeros(config['num_experiments'])
    dict_exp['thetas'] = np.zeros(config['num_experiments'])
    dict_exp['iterations'] = np.zeros(config['num_experiments'])

    if args.model == "cnn":
        pixels_x = train_shape[1]
        pixels_y = train_shape[2]
        num_channels = train_shape[3]
        dict_exp['conv11_acc'] = np.zeros((config['num_experiments'], 3 * 3 * num_channels * args.cnn_size))
        dict_exp['conv12_acc'] = np.zeros((config['num_experiments'], 3 * 3 * args.cnn_size * args.cnn_size))
        dict_exp['conv21_acc'] = np.zeros((config['num_experiments'], 3 * 3 * args.cnn_size * args.cnn_size))
        dict_exp['conv22_acc'] = np.zeros((config['num_experiments'], 3 * 3 * args.cnn_size * args.cnn_size))
        dict_exp['conv31_acc'] = np.zeros((config['num_experiments'], 3 * 3 * args.cnn_size * args.cnn_size))
        dict_exp['conv32_acc'] = np.zeros((config['num_experiments'], 3 * 3 * args.cnn_size * args.cnn_size))
        dict_exp['fc1_acc'] = np.zeros((config['num_experiments'], int((int(int((pixels_x + 1) / 2) + 1) / 2 + 1) / 2) * int(
            (int(int((pixels_y + 1) / 2) + 1) / 2 + 1) / 2) * args.cnn_size * args.fc_size))
        dict_exp['fc2_acc'] = np.zeros((config['num_experiments'], args.fc_size * 10))

    return dict_exp


def update_dict(dict_exp, args, sess, model, test_dict, experiment):

    if args.model == "ff":
        dict_exp['W1_acc'][experiment] = sess.run(model.W1).reshape(-1)
        dict_exp['W2_acc'][experiment] = sess.run(model.W2).reshape(-1)
        dict_exp['W3_acc'][experiment] = sess.run(model.W3).reshape(-1)

    elif args.model == "cnn":

        dict_exp['conv11_acc'][experiment] = sess.run(model.conv11.kernel).reshape(-1)
        dict_exp['conv12_acc'][experiment] = sess.run(model.conv12.kernel).reshape(-1)
        dict_exp['conv21_acc'][experiment] = sess.run(model.conv21.kernel).reshape(-1)
        dict_exp['conv22_acc'][experiment] = sess.run(model.conv22.kernel).reshape(-1)
        dict_exp['conv31_acc'][experiment] = sess.run(model.conv31.kernel).reshape(-1)
        dict_exp['conv32_acc'][experiment] = sess.run(model.conv32.kernel).reshape(-1)
        dict_exp['fc1_acc'][experiment] = sess.run(model.fc1.kernel).reshape(-1)
        dict_exp['fc2_acc'][experiment] = sess.run(model.fc2.kernel).reshape(-1)

    dict_exp['logits_acc'][experiment] = sess.run(model.logits, feed_dict=test_dict)
    dict_exp['preds'][experiment] = sess.run(model.y_pred, feed_dict=test_dict)

    return dict_exp