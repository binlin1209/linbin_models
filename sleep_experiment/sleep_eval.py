# -*- coding: utf-8 -*-

import time
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import sleep_inference
import sleep_train
from sleep_inputdata import *

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(sleep):
    with tf.Graph().as_default() as g:
#        x = tf.placeholder(tf.float32, [None, sleep_inference.INPUT_NODE], name='x-input')
#        y_ = tf.placeholder(tf.float32, [None, sleep_inference.OUTPUT_NODE], name='y-input')
        
        
        x = tf.placeholder(tf.float32, [
        100,
        sleep_inference.IMAGE_SIZE,
        sleep_inference.IMAGE_SIZE,
        sleep_inference.NUM_CHANNELS],
                       name='x-input')
        y_ = tf.placeholder(tf.float32, [None, sleep_inference.OUTPUT_NODE], name='y-input')
 
        validate_feed = {x: sleep.validation.images, y_: sleep.validation.labels}
        import pdb; pdb.set_trace()
        y = sleep_inference.inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(sleep_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(sleep_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    sleep = read_data_sets("sleep", one_hot=True)
    evaluate(sleep)

if __name__ == '__main__':
    main()