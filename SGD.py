# =========================================================================
# This code built for MCEN90048 project
# Written by Zhuo LI
# The Univerivsity of Melbourne
# zhuol7@student.unimelb.edu.au
# =========================================================================


import tensorflow as tf
import os
import json
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline


class TimeLiner:
    _timeline_dict = None

    def update_timeline(self, old_chrome_trace):
        # convert chrome trace to python dict
        chrome_trace_dict = json.loads(old_chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


data = input_data.read_data_sets('data/fashion', one_hot=True)
BATCH_SIZE = 100
TRAIN_DATA_SIZE = 55000
EPOCH = int(TRAIN_DATA_SIZE / BATCH_SIZE)  # 55000 / 100 = 550
X_VAL = data.validation.images
Y_VAL = data.validation.labels
X_TEST = data.test.images
Y_TEST = data.test.labels
LEARNING_RATE = 0.5
ckpt_path = 'results/MLPBaseline/SGD/SGD.ckpt'

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])


# fc1 layer
h_fc1 = tf.layers.dense(xs, 1024, tf.nn.relu)  # hidden layer
tf.summary.histogram('/h_fc1_output', h_fc1)
# fc2 layer
prediction = tf.layers.dense(h_fc1, 10)  # output layer
tf.summary.histogram('/prediction', prediction)

# the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)  # loss
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# merge all the summaries
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("results/MLPBaseline/SGD/", sess.graph)

saver = tf.train.Saver()

# firstly check if there exit model to be restored
# if no, start training. So if Richard wants to run in this training condition, plz delete the existing files first.
# else, restore model and test it on 'TEST DATA SET', and will just save another event file
if not os.path.exists('results/MLPBaseline/SGD/checkpoint'):  # if no model to restore
    print("no model to restore, start training.")
    sess.run(init)

    EPOCHS = EPOCH * 20
    for step in range(EPOCHS):
        batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)
        sess.run(train_step,
                 feed_dict={xs: batch_xs, ys: batch_ys},
                 options=None,
                 run_metadata=None)
        if (step+1) % 500 == 0:
            current_accuracy = sess.run(accuracy, feed_dict={xs: X_VAL, ys: Y_VAL})
            train_result = sess.run(merged, feed_dict={xs: X_VAL, ys: Y_VAL})
            writer.add_summary(train_result, step+1)
            print("step {}:\tTraining Accuracy={:.4f}".format(step+1, current_accuracy))

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    many_runs_timeline = TimeLiner()
    for step in range(5):
        batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)
        sess.run(train_step,
                 feed_dict={xs: batch_xs, ys: batch_ys},
                 options=options,
                 run_metadata=run_metadata)
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        many_runs_timeline.update_timeline(chrome_trace)
    many_runs_timeline.save('results/MLPBaseline/SGD/timeline_merged_of_%d_iterations.json' % 5)
    address = saver.save(sess, ckpt_path)
    print("Finish training, we'v stored the model's parameters to", address)
    final_accuracy = sess.run(accuracy, feed_dict={xs: X_TEST, ys: Y_TEST})
    print('test accuracy: %.4f' % final_accuracy)
    sess.close()
else:
    # restore the parameters and test the model
    saver.restore(sess, ckpt_path)
    print("Model restore successfully.")
    final_accuracy = sess.run(accuracy, feed_dict={xs: X_TEST, ys: Y_TEST})
    print('test accuracy: %.4f' % final_accuracy)
    sess.close()

