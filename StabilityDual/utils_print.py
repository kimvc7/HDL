from datetime import datetime
import numpy as np
import csv
from utils import total_gini
import tensorflow.compat.v1 as tf

def print_metrics(sess, model, nat_dict, val_dict, test_dict, ii, args, summary_writer, global_step):
    nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
    val_acc = sess.run(model.accuracy, feed_dict=val_dict)
    nat_xent = sess.run(model.xent, feed_dict=nat_dict)
    dual_xent = sess.run(model.dual_xent, feed_dict=nat_dict)
    MC_xent = sess.run(model.MC_xent, feed_dict=nat_dict)
    robust_xent = sess.run(model.robust_xent, feed_dict=nat_dict)
    robust_stable_xent = sess.run(model.robust_stable_xent, feed_dict=nat_dict)
    theta = sess.run(model.theta, feed_dict=nat_dict)

    print('Step {}:    ({})'.format(ii, datetime.now()))
    print('    training nat accuracy {:.4}'.format(nat_acc * 100))
    print('    validation nat accuracy {:.4}'.format(val_acc * 100))
    print('    Nat Xent {:.4}'.format(nat_xent))

    if args.stable:
        print('    theta {:.4}'.format(theta))
        print('    Stable xent upper bound with dual {:.4}'.format(dual_xent))
        print('    Stable Xent lower bound with Monte Carlo {:.4}'.format(MC_xent))

    if args.robust > 0 and args.model == "ff":
        print('    Robust Xent {:.4}'.format(robust_xent))
        if args.stable:
            print('    Robust Stable Xent {:.4}'.format(robust_stable_xent))

    if args.l0 > 0 and args.model == "ff":
        print('    W1_masked features', sum(sess.run(model.W1_masked).reshape(-1) > 0))
        print('    W2_masked features', sum(sess.run(model.W2_masked).reshape(-1) > 0))
        print('    W3_masked features', sum(sess.run(model.W3_masked).reshape(-1) > 0))
    if args.model == "ff":
        regularizer = sess.run(model.regularizer, feed_dict=nat_dict)
        print('    Regularizer', regularizer)

    # print("h1 std", np.std(sess.run(model.h1, feed_dict=nat_dict)))
    # print("h1 std 2", np.std(sess.run(model.h2, feed_dict=nat_dict), 0))
    # print('    W1 features', sum(sess.run(model.W1).reshape(-1) > 1e-12))
    # print('    W2 features', sum(sess.run(model.W2).reshape(-1) > 1e-12))
    # print('    W3 features', sum(sess.run(model.W3).reshape(-1) > 1e-12))

    summary1 = tf.Summary(value=[tf.Summary.Value(tag='TrainAcc', simple_value=nat_acc), ])
    summary2 = tf.Summary(value=[tf.Summary.Value(tag='ValAcc', simple_value=val_acc), ])
    summary3 = tf.Summary(value=[tf.Summary.Value(tag='TrainXent', simple_value=nat_xent), ])
    summary4 = tf.Summary(value=[tf.Summary.Value(tag='DualXent', simple_value=dual_xent), ])
    # summary4 = tf.Summary(value=[tf.Summary.Value(tag='Hidden_weights', simple_value=tf.summary.histogram('Hidden_weights', sess.run(model.W3).reshape(-1))),])
    # summary4 = tf.summary.histogram('Hidden_weights', model.W3.value())
    summary_writer.add_summary(summary1, global_step.eval(sess))
    summary_writer.add_summary(summary2, global_step.eval(sess))
    summary_writer.add_summary(summary3, global_step.eval(sess))
    summary_writer.add_summary(summary4, global_step.eval(sess))
    # summary_writer.add_text('args', str(args), global_step.eval(sess))
    summary5 = sess.run(model.summary, feed_dict=test_dict)
    summary_writer.add_summary(summary5, global_step.eval(sess))

    return val_acc

def update_dict_output(dict_exp, experiment, sess, test_acc, model, test_dict, num_iters):
    dict_exp['test_accs'][experiment] = test_acc*100
    dict_exp['thetas'][experiment] = sess.run(model.theta, feed_dict=test_dict)
    dict_exp['iterations'][experiment] = num_iters

    return dict_exp


def print_stability_measures(dict_exp, args, num_experiments, batch_size, subset_ratio, avg_test_acc, max_num_training_steps):

    avg_test_acc = avg_test_acc / num_experiments
    print('  Average testing accuracy {:.4}'.format(avg_test_acc * 100))
    print('  Theta values', dict_exp['thetas'])
    # print('  individual accuracies: \n', test_accs)
    std = np.array([float(k) for k in dict_exp['test_accs']]).std()
    print('Test Accuracy std {:.2}'.format(np.array([float(k) for k in dict_exp['test_accs']]).std()))
    print("Logits std", np.mean(np.mean(np.std(dict_exp['logits_acc'], axis=0), axis=0)))
    logit_stability = np.mean(np.std(dict_exp['logits_acc'], axis=0), axis=0)
    gini_stability = total_gini(dict_exp['preds'].transpose())
    print("Gini stability", gini_stability)



    if args.model == "ff":
        w1_stability, w2_stability, w3_stability = print_layer_stability_ff(dict_exp)
    elif args.model == "cnn":
        conv11_stability, conv12_stability, conv21_stability, conv22_stability, conv31_stability, conv32_stability, fc1_stability, fc2_stability = print_layer_stability_cnn(dict_exp)

    file = open(str('results_' + args.model + args.data_set + '.csv'), 'a+', newline='')
    with file:
        writer = csv.writer(file)
        if args.model == "ff":
            writer.writerow(
                [args.stable, args.robust, num_experiments, args.train_size, batch_size, subset_ratio, avg_test_acc, dict_exp['test_accs'], std,
                dict_exp['thetas'], max_num_training_steps, dict_exp['iterations'], w1_stability, w2_stability, w3_stability, logit_stability,
                gini_stability, args.l2, args.l0])
        elif args.model == "cnn":
            writer.writerow(
                [args.stable, args.robust, num_experiments, args.train_size, batch_size, subset_ratio, avg_test_acc, dict_exp['test_accs'], std,
                 dict_exp['thetas'], max_num_training_steps, dict_exp['iterations'], conv11_stability, conv12_stability, conv21_stability,
                 conv22_stability, conv31_stability, conv32_stability, fc1_stability, fc2_stability, logit_stability,
                 gini_stability, args.l2, args.l0, args.robust])


def print_layer_stability_ff(dict_exp):
    w1_stability = np.mean(np.std(dict_exp['W1_acc'], axis=0), axis=0)
    w2_stability = np.mean(np.std(dict_exp['W2_acc'], axis=0), axis=0)
    w3_stability = np.mean(np.std(dict_exp['W3_acc'], axis=0), axis=0)
    print("W1 std", w1_stability)
    print("W2 std", w2_stability)
    print("W3 std", w3_stability)
    return w1_stability, w2_stability, w3_stability

def print_layer_stability_cnn(dict_exp):
    conv11_stability = np.mean(np.std(dict_exp['conv11_acc'], axis=0), axis=0)
    conv12_stability = np.mean(np.std(dict_exp['conv12_acc'], axis=0), axis=0)
    conv21_stability = np.mean(np.std(dict_exp['conv21_acc'], axis=0), axis=0)
    conv22_stability = np.mean(np.std(dict_exp['conv22_acc'], axis=0), axis=0)
    conv31_stability = np.mean(np.std(dict_exp['conv31_acc'], axis=0), axis=0)
    conv32_stability = np.mean(np.std(dict_exp['conv31_acc'], axis=0), axis=0)
    fc1_stability = np.mean(np.std(dict_exp['fc1_acc'], axis=0), axis=0)
    fc2_stability = np.mean(np.std(dict_exp['fc2_acc'], axis=0), axis=0)

    print("conv11 std", conv11_stability)
    print("conv12 std", conv12_stability)
    print("conv21 std", conv21_stability)
    print("conv22 std", conv22_stability)
    print("conv31 std", conv31_stability)
    print("conv32 std", conv32_stability)
    print("fc1 std", fc1_stability)
    print("fc2 std", fc2_stability)

    return conv11_stability, conv12_stability, conv21_stability, conv22_stability,conv31_stability,conv32_stability, fc1_stability, fc2_stability


# Initialize the summary writer, global variables, and our time counter.
# summary_writer = tf.summary.FileWriter(model_dir + "/Xent")
# summary_writer1 = tf.summary.FileWriter(model_dir+ "/Max_Xent")
# summary_writer2 = tf.summary.FileWriter(model_dir+ "/Accuracy")
# summary_writer3 = tf.summary.FileWriter(model_dir+ "/Test_Accuracy")

# Tensorboard Summaries
# summary = tf.Summary(value=[
#    tf.Summary.Value(tag='Xent', simple_value= nat_xent),])
# summary1 = tf.Summary(value=[
#    tf.Summary.Value(tag='Xent', simple_value= max_xent),])
# summary2 = tf.Summary(value=[
#    tf.Summary.Value(tag='Accuracy', simple_value= nat_acc*100)])
# summary3 = tf.Summary(value=[
#    tf.Summary.Value(tag='Accuracy', simple_value= test_acc*100)])
# summary_writer.add_summary(summary, global_step.eval(sess))
# summary_writer1.add_summary(summary1, global_step.eval(sess))
# summary_writer2.add_summary(summary2, global_step.eval(sess))
# summary_writer3.add_summary(summary3, global_step.eval(sess))


# Initialize the summary writer, global variables, and our time counter.
# summary_writer = tf.summary.FileWriter(model_dir + "/Xent")
# summary_writer1 = tf.summary.FileWriter(model_dir+ "/Max_Xent")
# summary_writer2 = tf.summary.FileWriter(model_dir+ "/Accuracy")
# summary_writer3 = tf.summary.FileWriter(model_dir+ "/Test_Accuracy")


# Tensorboard Summaries
# summary = tf.Summary(value=[
#    tf.Summary.Value(tag='Xent', simple_value= nat_xent),])
# summary1 = tf.Summary(value=[
#    tf.Summary.Value(tag='Xent', simple_value= max_xent),])
# summary2 = tf.Summary(value=[
#    tf.Summary.Value(tag='Accuracy', simple_value= nat_acc*100)])
# summary3 = tf.Summary(value=[
#    tf.Summary.Value(tag='Accuracy', simple_value= test_acc*100)])
# summary_writer.add_summary(summary, global_step.eval(sess))
# summary_writer1.add_summary(summary1, global_step.eval(sess))
# summary_writer2.add_summary(summary2, global_step.eval(sess))
# summary_writer3.add_summary(summary3, global_step.eval(sess))