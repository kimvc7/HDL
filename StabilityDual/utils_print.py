from datetime import datetime

def print_metrics(sess, model, nat_dict, val_dict, ii, args):
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

    return val_acc

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