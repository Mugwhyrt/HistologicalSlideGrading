#
#   Nth Level Slide ROI Classifier, 50x50
# --------------------------------------------
# Classifies ROI for 50x50 image patches of H&E stained histology slides
# Slides must be at the top level (lowest level of zoom)
#
# @Zachary Rohman, Fall 2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensor_vis as tv
from PIL import Image
import random_tensor as rt
import random, os
#import matplotlib.pyplot as plt

MEAN = 127.5
STD = 73.9002685546875

DATA_PATH = "D:/Grading_slides/0_level/255x255/"
MODEL_PATH = "D:/Grading_slides/0_level/255_grading_net/"

DROPOUT = 0.5
# for each epoch: 0.01 * 0.95 ^(epoch number)
# In Epoch 1: Learning rate bumped down from 0.095 -> 0.0095 to counteract NaN loss
LEARNING_RATE = 0.000046329

tf.logging.set_verbosity(tf.logging.INFO)
tf.contrib.eager.seterr(inf_or_nan="ignore")

def cnn_model_fn(features, labels, mode):
    #Input layer
    #reshape X to 4_D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 255, 255, 3])

    #Conv. layer 1
    #input tensor shape: [batch_size, 50, 50, 3]
    #output tensor shape: [batch_size, 50, 50, 96] (maybe)
    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 64,
        kernel_size = [11, 11],
        strides = (4,4),
        padding = "same",
        activation = tf.nn.relu,
        trainable = True)

    #Pooling layer 1
    #input dimensions: [batch_size, 50, 50, 96]
    #output dimensions: [batch_size, 25, 25, 96]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    #Conv. layer 2
    #input shape: [batch_size, 25, 25, 96]
    #output shape: [batch size, 25, 25, 192]
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 128,
        kernel_size = [7,7],
        strides = (3,3),
        padding = "same",
        activation = tf.nn.relu)

    #Pooling layer 2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size= [2,2],
                                    strides = 2)

    conv3 = tf.layers.conv2d(
        inputs = pool2,
        filters = 256,
        kernel_size = [5,5],
        strides = (2,2),
        padding = "same",
        activation = tf.nn.relu)

    #Pooling layer 2
    #pool3 = tf.layers.max_pooling2d(inputs=conv3,
    #                                pool_size= [2,2],
    #                                strides = 2)
    conv4 = tf.layers.conv2d(
        inputs = conv3,
        filters = 512,
        kernel_size = [3,3],
        strides = (1,1),
        padding = "same",
        activation = tf.nn.relu)

    #conv5 = tf.layers.conv2d(
    #    inputs = conv4,
    #    filters = 32,
    #    kernel_size = [5,5],
    #    padding = "same",
    #    activation = tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv4,
                                    pool_size= [2,2],
                                    strides = 2)
    
    
    #Flatten tensor into a batch of vectors
    pool4_flat = tf.reshape(pool4,
                            [-1, pool4.shape[1] * pool4.shape[2] * pool4.shape[3]])

    dense1 = tf.layers.dense(inputs = pool4_flat,
                            units=512,
                            activation=tf.nn.relu)

    dense2 = tf.layers.dense(inputs = dense1,
                            units=1024,
                            activation=tf.nn.relu)
    dense3 = tf.layers.dense(inputs = dense2,
                            units=512,
                            activation=tf.nn.relu)
    #dense4 = tf.layers.dense(inputs = dense3,
    #                        units=512,
    #                        activation=tf.nn.relu)
    print("/ndense3 is {}".format(dense3.shape))
    dropout = tf.layers.dropout( inputs = dense3, rate= DROPOUT,
                                 training = mode == tf.estimator.ModeKeys.TRAIN)
    print("/ndropout is {}/n".format(dropout.shape))
    #Logits Layer
    logits = tf.layers.dense(inputs = dropout, units = 3)
    
    predictions = {
        #Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        #Add 'softmax_tensor' to the graph. It is used for PREDICT and by
        #the 'logging hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions = predictions)

    print("/nlogits is: {}/n".format(logits.shape))
    # calculate loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(logits = logits,
                                           onehot_labels = tf.one_hot(labels, 3))
    
    #loss_arr = tf.map_fn(tf.make_ndarray, loss)
    #print("loss is . . .")
    #for i in tf.map_fn(loss):
    #    print(i)
    #print("loss has NaN:\n{}".format(np.isnan(loss_arr)))
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = LEARNING_RATE)
        train_op = optimizer.minimize(
            loss = loss,
            global_step=tf.train.get_global_step())
        print("calculations done, returning EstimatorSpec()")
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)
    print("must be in EVAL mode!")
    #Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels = labels, predictions = predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(unused_argv):
    Train_Model = False
    train_steps = 10000
    Retrain_Model = False
    retrain_steps = 500
    Eval_Model = False
    Predict_Model = True
    Random_Data = False
    eval_size = 20
    train_size = 1000

    eval_data = np.load(DATA_PATH + "eval_data.npy")[:, :, :, 0:3]
    eval_labels = np.asarray(np.load(DATA_PATH + "eval_labels.npy"), dtype ="int32")
    
    roi_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir = MODEL_PATH)
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors = tensors_to_log, every_n_iter = 50)
    if Random_Data:
        #eval_data = rt.randomized_tensor(eval_data)
        train_data = rt.randomized_tensor(train_data)
    ##############################
    #       TRAIN MODEL          # 
    ##############################
    #print("test_data: {}".format(type(train_data)))
    #print("test_data: {}".format(train_data.shape))
    if Train_Model:
        train_data = np.load(DATA_PATH + "train_data.npy")[:, :, :, 0:3]
        train_labels = np.load(DATA_PATH + "train_labels.npy")
        s = random.randrange(len(train_data) - train_size)
        train_data = train_data[s:s+train_size]
        train_labels = train_labels[s:s+train_size]
        print("loading grading data . . .")
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x":train_data},
            y = train_labels,
            batch_size = 20,
            num_epochs = None,
            shuffle = True)
        print("training grading estimator . . .")
        roi_classifier.train(
            input_fn = train_input_fn,
            steps = train_steps,
            hooks = [logging_hook])
    #############################
    #       EVAL MODEL          #
    #############################
    if Eval_Model:
        #Open Mean normalized data and label arrays
        eval_data = np.load(DATA_PATH + "eval_data.npy")[:, :, :, 0:3]
        eval_labels = np.asarray(np.load(DATA_PATH + "eval_labels.npy"), dtype ="int32")
        for i in range(10):
            s = random.randrange(len(eval_data) - eval_size)
            temp_eval_data = eval_data[s:s+eval_size, :, :, 0:3]
            temp_eval_labels = eval_labels[s:s+eval_size]

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": temp_eval_data},
                y=temp_eval_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = roi_classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)
##################################################
#           PREDICT MODEL                        #
##################################################
    if Predict_Model:
        # ['conv2d/bias', 'conv2d/kernel', 'conv2d_1/bias',
        #'conv2d_1/kernel', 'dense/bias', 'dense/kernel',
        #'dense_1/bias', 'dense_1/kernel', 'global_step']
        #tv.plot_conv_weights(roi_classifier.get_variable_value('conv2d/kernel'))
        #tv.plot_conv_weights(roi_classifier.get_variable_value('conv2d_1/kernel'))
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        pred_results = roi_classifier.predict(input_fn=pred_input_fn)

        preds = []
        error_labels = []
        error_data = []
        error_true = []
        TP = 0.0
        FP = 0.0
        FN = 0.0
        for p in enumerate(pred_results):
            preds.append(p[1]['classes'])
            if p[1]['classes'] != eval_labels[p[0]]:
                if p[1]['classes'] == 0:
                    FN += 1
                else:
                    FP += 1
                error_labels.append(p[1]['classes'])
                error_data.append(eval_data[p[0]])
                error_true.append(eval_labels[p[0]])
            elif p[1]['classes'] == 1:
                TP += 1
        print("TP: {}\tFP: {}\tFN: {}".format(TP, FP, FN))
        print("Precision: {}\tRecall: {}".format((TP / (TP + FP)),
                                                 (TP / (TP + FN))))
        error_data = np.asarray(error_data)
        """
        s = int(random.random() * len(eval_labels) - 10)
        e = s + 9
        s = 0
        e = 9
        #print("eval_data[s:e]: {}\n".format(eval_data[s:e]))
        #print("eval_labels[s:e]: {}\n".format(eval_labels[s:e]))
        #print("preds[s:e]: {}\n".format(preds[s:e]))
        eval_data = (eval_data * STD) + MEAN
        tv.plot_images(eval_data[s:e],eval_labels[s:e], preds[s:e])
        s = int(random.random() * len(error_labels) - 10)
        e = s + 9
        error_data = (error_data * STD) + MEAN
        tv.plot_images(error_data[s:e],error_true[s:e], error_labels[s:e])
    
        """
                        
if __name__ == "__main__":
    tf.app.run()
                             
