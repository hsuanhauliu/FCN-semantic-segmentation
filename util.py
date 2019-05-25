"""
    This is a program written for CSCI677 Advanced Computer Vision homework
    assignment 6.
"""

import glob
import os
import pickle
import random

import matplotlib
matplotlib.use('agg')

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def read_data(image_dir_path, label_dir_path, gt_dir_path):
    """ Read images and their labels from the specified path """
    images = []
    labels = []
    gts = []
    image_extension = '.png'
    label_extension = '.label'

    for i in glob.glob(image_dir_path + '/*' + image_extension):
        data_name = i.split(image_dir_path)[1].split(image_extension)[0]
        breakdown = data_name.split('_')
        label_name = '_'.join([breakdown[0], 'road', breakdown[1]]) + label_extension
        label_path = os.path.join(label_dir_path, label_name)
        gt_name = '_'.join([breakdown[0], 'road', breakdown[1]]) + image_extension
        gt_path = os.path.join(gt_dir_path, gt_name)

        # load the image
        img = cv.imread(i)
        images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))

        # load the ground truths
        gt = cv.imread(gt_path)
        gts.append(cv.cvtColor(gt, cv.COLOR_BGR2RGB))

        # load the label
        with open(label_path, 'rb') as in_file:
            label = pickle.load(in_file, encoding='bytes')
            labels.append(label)

    return np.array(images), np.array(labels), np.array(gts)


def split_train_and_validation(data, split):
    """ Split the entire data into training and validation set """
    return data[:split], data[split:]


def generate_random_index(size):
    """ Randomly generate an index """
    return random.randint(0, size - 1)


def calculate_iou(y_predict_masked, y_masked):
    """ Calculate IoU of road label (class 1) """
    y_predict_label = tf.greater(y_predict_masked, 0.5)  # contain true and false
    y_real_label = tf.equal(y_masked, 1)  # contain true and false
    tp = tf.to_float(tf.count_nonzero(tf.logical_and(y_predict_label, y_real_label)))
    diff = tf.to_float(tf.count_nonzero(tf.logical_xor(y_predict_label, y_real_label)))

    return tf.divide(tp, tf.add(tp, diff))


def output_image(img, prediction, gt, name, step):
    """ Map prediction labels to their corresponding colors and output the image """
    prediction = np.squeeze(prediction)
    output_img = np.zeros((prediction.shape[0], prediction.shape[1], 3)).astype(int)
    output_img[prediction] = [255, 0, 255]
    output_img[np.logical_not(prediction)] = [255, 0, 0]
    separator = np.zeros((5, prediction.shape[1], 3), dtype=int)
    output_img = np.vstack((img, separator, output_img, separator, gt))

    plt.imshow(output_img)
    plt.savefig('results/{}-result-{}.png'.format(name, step), bbox_inches='tight')
    plt.close()


def fcn_32(data):
    """ Fully Convolutional Network FCN-32 structure """
    bias = True
    bias_init = tf.contrib.layers.xavier_initializer()
    kernel_init = tf.contrib.layers.xavier_initializer()

    y_1 = tf.layers.conv2d(
        name='ConvLayer1',
        inputs=data,
        filters=64,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_2 = tf.layers.conv2d(
        name='ConvLayer2',
        inputs=y_1,
        filters=64,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_2_max = tf.layers.max_pooling2d(
        name='MaxPooling1',
        padding='same',
        inputs=y_2, pool_size=2, strides=2)

    y_3 = tf.layers.conv2d(
        name='ConvLayer3',
        inputs=y_2_max,
        filters=128,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_4 = tf.layers.conv2d(
        name='ConvLayer4',
        inputs=y_3,
        filters=128,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_4_max = tf.layers.max_pooling2d(
        name='MaxPooling2',
        padding='same',
        inputs=y_4, pool_size=2, strides=2)

    y_5 = tf.layers.conv2d(
        name='ConvLayer5',
        inputs=y_4_max,
        filters=256,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_6 = tf.layers.conv2d(
        name='ConvLayer6',
        inputs=y_5,
        filters=256,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_7 = tf.layers.conv2d(
        name='ConvLayer7',
        inputs=y_6,
        filters=256,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_7_max = tf.layers.max_pooling2d(
        name='MaxPooling3',
        padding='same',
        inputs=y_7, pool_size=2, strides=2)

    y_8 = tf.layers.conv2d(
        name='ConvLayer8',
        inputs=y_7_max,
        filters=512,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_9 = tf.layers.conv2d(
        name='ConvLayer9',
        inputs=y_8,
        filters=512,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_10 = tf.layers.conv2d(
        name='ConvLayer10',
        inputs=y_9,
        filters=512,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_10_max = tf.layers.max_pooling2d(
        name='MaxPooling4',
        padding='same',
        inputs=y_10, pool_size=2, strides=2)

    y_11 = tf.layers.conv2d(
        name='ConvLayer11',
        inputs=y_10_max,
        filters=512,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_12 = tf.layers.conv2d(
        name='ConvLayer12',
        inputs=y_11,
        filters=512,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_13 = tf.layers.conv2d(
        name='ConvLayer13',
        inputs=y_12,
        filters=512,
        kernel_size=3,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_13_max = tf.layers.max_pooling2d(
        name='MaxPooling5',
        padding='same',
        inputs=y_13, pool_size=2, strides=2)

    y_14 = tf.layers.conv2d(
        name='ConvLayer14',
        inputs=y_13_max,
        filters=4096,
        kernel_size=7,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_15 = tf.layers.conv2d(
        name='ConvLayer15',
        inputs=y_14,
        filters=4096,
        kernel_size=1,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_16 = tf.layers.conv2d(
        name='ConvLayer16',
        inputs=y_15,
        filters=1,
        kernel_size=1,
        padding='same',
        use_bias=bias,
        activation=tf.nn.relu,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    y_17 = tf.layers.conv2d_transpose(
        name='DeConvLayer1',
        inputs=y_16,
        filters=1,
        kernel_size=64,
        strides=32,
        padding='same',
        use_bias=bias,
        bias_initializer=bias_init,
        kernel_initializer=kernel_init)

    return y_17
