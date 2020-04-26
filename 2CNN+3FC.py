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
LEARNING_RATE = 0.001
ckpt_path = 'results/CNNModel/2CNN+3FC/2CNN+3FC.ckpt'


def conv_layer(input_value, size_in, size_out, kernel_size, strides, name):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, size_in, size_out], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input_value, w, strides=[1, strides, strides, 1], padding="SAME")
        conv_output = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram(name+"output", conv_output)
        return conv_output


def fc_layer(input_value, size_in, size_out, name, use_relu=True):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.05), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        fc_output = tf.matmul(input_value, w) + b
        if use_relu:
            fc_output = tf.nn.relu(fc_output)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram(name+"output", fc_output)
        return fc_output


# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784]) / 255.  # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('reshape'):
    x_image = tf.reshape(xs, [-1, 28, 28, 1])  # last '1' is channel, 1 means black and white

tf.summary.image('the last 5 images being processed', x_image, 5)

# CNN
# shape (28, 28, 1) -> (28, 28, 32)
conv1 = conv_layer(x_image, 1, 32, 5, 1, 'conv1')
# -> (14, 14, 32)
pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, name='pooling1')
#  (14, 14, 32) -> (14, 14, 64)
conv2 = conv_layer(pool1, 32, 64, 5, 1, 'conv2')
# -> (7, 7, 64)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pooling2')

# flatting, [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
with tf.name_scope('flatting'):
    h_pool2_flat = tf.reshape(pool2, [-1, 7*7*64])

# fc layer 1: hidden layer1
h_fc1 = fc_layer(h_pool2_flat, 7*7*64, 1024, 'hidden_layer1')

# fc layer 2: hidden layer2
h_fc2 = fc_layer(h_fc1, 1024, 1024, 'hidden_layer2')

# fc layer 3: output layer
prediction = fc_layer(h_fc2, 1024, 10, 'output_layer', use_relu=False)
tf.summary.histogram('/prediction', prediction)

# the error between prediction and real data
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)  # loss
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE, epsilon=1e-8).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# merge all the summaries
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("results/CNNModel/2CNN+3FC/", sess.graph)

saver = tf.train.Saver()

# firstly check if there exit model to be restored
# if no, start training. So if Richard wants to run in this training condition, plz delete the existing files first.
# else, restore model and test it on 'TEST DATA SET', and will just save another event file
if not os.path.exists('results/CNNModel/2CNN+3FC/checkpoint'):  # if no model to restore
    print("no model to restore, start training.")
    sess.run(init)

    EPOCHS = EPOCH * 6
    for step in range(EPOCHS):
        batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)
        sess.run(train_step,
                 feed_dict={xs: batch_xs, ys: batch_ys},
                 options=None,
                 run_metadata=None)
        if (step+1) % 200 == 0:
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
    many_runs_timeline.save('results/CNNModel/2CNN+3FC/timeline_merged_of_%d_iterations.json' % 5)
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
