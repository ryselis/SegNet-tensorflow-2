import csv
import functools
import itertools
import json
import os
import random
import time
from datetime import datetime, timedelta

import numpy as np
import tensorflow as tf
from scipy import misc
from skimage.io import imsave
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.metrics import Precision, Accuracy, Recall
from tensorflow.python.training.adam import AdamOptimizer

from dataset import get_dataset_generator, get_full_data_generator
from drawings_object import draw_plots_bayes, draw_plots_bayes_external
from evaluation_object import cal_loss, normal_loss, per_class_acc, get_hist, print_hist_summary, train_op, MAX_VOTE, \
    var_calculate, weighted_loss
from inputs_object import get_filename_list, get_all_test_data, get_dataset
from layers_object import conv_layer, up_sampling, max_pool, initialization, \
    variable_with_weight_decay, ConvLayerCompat


class SegNetCompatModel(tf.keras.layers.Layer):
    def __init__(self, conf_file="config.json", is_training=True, with_dropout=True, keep_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        with open(conf_file) as f:
            self.config = json.load(f)

        self.num_classes = self.config["NUM_CLASSES"]
        self.use_vgg = self.config["USE_VGG"]

        if self.use_vgg is False:
            self.vgg_param_dict = None
            print("No VGG path in config, so learning from scratch")
        else:
            self.vgg16_npy_path = self.config["VGG_FILE"]
            self.vgg_param_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
            print("VGG parameter loaded")

        self.train_file = self.config["TRAIN_FILE"]
        self.val_file = self.config["VAL_FILE"]
        self.test_file = self.config["TEST_FILE"]
        self.img_prefix = self.config["IMG_PREFIX"]
        self.label_prefix = self.config["LABEL_PREFIX"]
        self.bayes = self.config["BAYES"]
        self.opt = self.config["OPT"]
        self.saved_dir = self.config["SAVE_MODEL_DIR"]
        self.input_w = self.config["INPUT_WIDTH"]
        self.input_h = self.config["INPUT_HEIGHT"]
        self.input_c = self.config["INPUT_CHANNELS"]
        self.tb_logs = self.config["TB_LOGS"]
        self.batch_size = self.config["BATCH_SIZE"]

        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_acc = [], []

        self.model_version = 0  # used for saving the model
        self.saver = None
        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None

        self.is_training = is_training
        self.with_dropout_pl = with_dropout
        self.keep_prob_pl = keep_prob

        self.input_layer = Input(shape=[self.input_h, self.input_w, self.input_c], dtype=tf.float32)

    @tf.compat.v1.keras.utils.track_tf1_style_variables
    def call(self, inputs):
        batch_size_pl = self.batch_size
        is_training_pl = self.is_training
        with_dropout_pl = self.with_dropout_pl
        keep_prob_pl = self.keep_prob_pl

        # Before enter the images into the architecture, we need to do Local Contrast Normalization
        # But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
        # Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization

        norm1 = tf.nn.lrn(inputs, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
        # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
        conv1_1 = ConvLayerCompat("conv1_1", [3, 3, 3, 64], is_training_pl, self.use_vgg, self.vgg_param_dict)(norm1)
        # conv1_1 = conv_layer(norm1, )
        conv1_2 = conv_layer(conv1_1, "conv1_2", [3, 3, 64, 64], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        pool1, pool1_index, shape_1 = max_pool(conv1_2, 'pool1')

        # Second box of convolution layer(4)
        conv2_1 = conv_layer(pool1, "conv2_1", [3, 3, 64, 128], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        conv2_2 = conv_layer(conv2_1, "conv2_2", [3, 3, 128, 128], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        pool2, pool2_index, shape_2 = max_pool(conv2_2, 'pool2')

        # Third box of convolution layer(7)
        conv3_1 = conv_layer(pool2, "conv3_1", [3, 3, 128, 256], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        conv3_2 = conv_layer(conv3_1, "conv3_2", [3, 3, 256, 256], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        conv3_3 = conv_layer(conv3_2, "conv3_3", [3, 3, 256, 256], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        pool3, pool3_index, shape_3 = max_pool(conv3_3, 'pool3')

        # Fourth box of convolution layer(10)
        if self.bayes:
            dropout1 = tf.compat.v1.layers.dropout(pool3, rate=(1 - keep_prob_pl),
                                                   training=with_dropout_pl, name="dropout1")
            conv4_1 = conv_layer(dropout1, "conv4_1", [3, 3, 256, 512], is_training_pl, self.use_vgg,
                                 self.vgg_param_dict)
        else:
            conv4_1 = conv_layer(pool3, "conv4_1", [3, 3, 256, 512], is_training_pl, self.use_vgg,
                                 self.vgg_param_dict)
        conv4_2 = conv_layer(conv4_1, "conv4_2", [3, 3, 512, 512], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        conv4_3 = conv_layer(conv4_2, "conv4_3", [3, 3, 512, 512], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        pool4, pool4_index, shape_4 = max_pool(conv4_3, 'pool4')

        # Fifth box of convolution layers(13)
        if self.bayes:
            dropout2 = tf.compat.v1.layers.dropout(pool4, rate=(1 - keep_prob_pl),
                                                   training=with_dropout_pl, name="dropout2")
            conv5_1 = conv_layer(dropout2, "conv5_1", [3, 3, 512, 512], is_training_pl, self.use_vgg,
                                 self.vgg_param_dict)
        else:
            conv5_1 = conv_layer(pool4, "conv5_1", [3, 3, 512, 512], is_training_pl, self.use_vgg,
                                 self.vgg_param_dict)
        conv5_2 = conv_layer(conv5_1, "conv5_2", [3, 3, 512, 512], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        conv5_3 = conv_layer(conv5_2, "conv5_3", [3, 3, 512, 512], is_training_pl, self.use_vgg,
                             self.vgg_param_dict)
        pool5, pool5_index, shape_5 = max_pool(conv5_3, 'pool5')

        # ---------------------So Now the encoder process has been Finished--------------------------------------#
        # ------------------Then Let's start Decoder Process-----------------------------------------------------#

        # First box of deconvolution layers(3)
        if self.bayes:
            dropout3 = tf.compat.v1.layers.dropout(pool5, rate=(1 - keep_prob_pl),
                                                   training=with_dropout_pl, name="dropout3")
            deconv5_1 = up_sampling(dropout3, pool5_index, shape_5, batch_size_pl,
                                    name="unpool_5")
        else:
            deconv5_1 = up_sampling(pool5, pool5_index, shape_5, batch_size_pl,
                                    name="unpool_5")
        deconv5_2 = conv_layer(deconv5_1, "deconv5_2", [3, 3, 512, 512], is_training_pl)
        deconv5_3 = conv_layer(deconv5_2, "deconv5_3", [3, 3, 512, 512], is_training_pl)
        deconv5_4 = conv_layer(deconv5_3, "deconv5_4", [3, 3, 512, 512], is_training_pl)
        # Second box of deconvolution layers(6)
        if self.bayes:
            dropout4 = tf.compat.v1.layers.dropout(deconv5_4, rate=(1 - keep_prob_pl),
                                                   training=with_dropout_pl, name="dropout4")
            deconv4_1 = up_sampling(dropout4, pool4_index, shape_4, batch_size_pl,
                                    name="unpool_4")
        else:
            deconv4_1 = up_sampling(deconv5_4, pool4_index, shape_4, batch_size_pl,
                                    name="unpool_4")
        deconv4_2 = conv_layer(deconv4_1, "deconv4_2", [3, 3, 512, 512], is_training_pl)
        deconv4_3 = conv_layer(deconv4_2, "deconv4_3", [3, 3, 512, 512], is_training_pl)
        deconv4_4 = conv_layer(deconv4_3, "deconv4_4", [3, 3, 512, 256], is_training_pl)
        # Third box of deconvolution layers(9)
        if self.bayes:
            dropout5 = tf.compat.v1.layers.dropout(deconv4_4, rate=(1 - keep_prob_pl),
                                                   training=with_dropout_pl, name="dropout5")
            deconv3_1 = up_sampling(dropout5, pool3_index, shape_3, batch_size_pl,
                                    name="unpool_3")
        else:
            deconv3_1 = up_sampling(deconv4_4, pool3_index, shape_3, batch_size_pl,
                                    name="unpool_3")
        deconv3_2 = conv_layer(deconv3_1, "deconv3_2", [3, 3, 256, 256], is_training_pl)
        deconv3_3 = conv_layer(deconv3_2, "deconv3_3", [3, 3, 256, 256], is_training_pl)
        deconv3_4 = conv_layer(deconv3_3, "deconv3_4", [3, 3, 256, 128], is_training_pl)
        # Fourth box of deconvolution layers(11)
        if self.bayes:
            dropout6 = tf.compat.v1.layers.dropout(deconv3_4, rate=(1 - keep_prob_pl),
                                                   training=with_dropout_pl, name="dropout6")
            deconv2_1 = up_sampling(dropout6, pool2_index, shape_2, batch_size_pl,
                                    name="unpool_2")
        else:
            deconv2_1 = up_sampling(deconv3_4, pool2_index, shape_2, batch_size_pl,
                                    name="unpool_2")
        deconv2_2 = conv_layer(deconv2_1, "deconv2_2", [3, 3, 128, 128], is_training_pl)
        deconv2_3 = conv_layer(deconv2_2, "deconv2_3", [3, 3, 128, 64], is_training_pl)
        # Fifth box of deconvolution layers(13)
        deconv1_1 = up_sampling(deconv2_3, pool1_index, shape_1, batch_size_pl,
                                name="unpool_1")
        deconv1_2 = conv_layer(deconv1_1, "deconv1_2", [3, 3, 64, 64], is_training_pl)
        deconv1_3 = conv_layer(deconv1_2, "deconv1_3", [3, 3, 64, 64], is_training_pl)

        with tf.compat.v1.variable_scope('conv_classifier') as scope:
            kernel = variable_with_weight_decay('weights', initializer=initialization(1, 64),
                                                shape=[1, 1, 64, self.num_classes], wd=False)
            conv = tf.nn.conv2d(input=deconv1_3, filters=kernel, strides=[1, 1, 1, 1],
                                padding='SAME')
            biases = variable_with_weight_decay('biases', tf.compat.v1.constant_initializer(0.0),
                                                shape=[self.num_classes], wd=False)
            logits = tf.nn.bias_add(conv, biases, name=scope.name)
        return logits

    def train_v2(self, batch_size: int):
        model = self(self.input_layer)
        tf_model = Model(self.input_layer, model, name='SegNet')
        loss_weight = np.array([
            0.2595,
            0.1826,
            4.5640,
            0.1417,
            0.9051,
            0.3826,
            9.6446,
            1.8418,
            0.6823,
            6.2478,
            7.3614,
            1.0974
        ])
        loss = functools.partial(weighted_loss, number_class=self.num_classes, frequency=loss_weight)
        tf_model.compile(optimizer=AdamOptimizer(learning_rate=0.001),
                         loss=loss,
                         metrics=[Precision(), Accuracy(), Recall()])

        image_filename, label_filename = get_filename_list(self.train_file, self.config)
        val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)

        dataset = get_dataset(image_filename, label_filename, config=self.config)
        validation_dataset = get_dataset(val_image_filename, val_label_filename, config=self.config)

        # self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, batch_size, self.config)
        # self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size,
        #                                                   self.config)
        tf_model.summary()
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints2',
                                                         save_weights_only=True,
                                                         verbose=1)
        history = tf_model.fit(x=dataset,
                               batch_size=batch_size,
                               epochs=1,
                               validation_data=validation_dataset,
                               validation_batch_size=batch_size,
                               callbacks=[cp_callback])
        print(history.history)


def is_male_file(file_name):
    male_participant_names = ["Andrius", "Ryselis", "Rytis", "Tautvydas", "Tomas"]
    if any(name in file_name for name in male_participant_names):
        return True

    female_participant_names = ["Reda", "Renata"]
    if any(name in file_name for name in female_participant_names):
        return False

    male_labels = [f"V{i}_" for i in range(1, 22)]
    return any(label in file_name for label in male_labels)


class SegNet:
    def __init__(self, conf_file="config.json"):
        with open(conf_file) as f:
            self.config = json.load(f)

        self.num_classes = self.config["NUM_CLASSES"]
        self.use_vgg = self.config["USE_VGG"]

        if self.use_vgg is False:
            self.vgg_param_dict = None
            print("No VGG path in config, so learning from scratch")
        else:
            self.vgg16_npy_path = self.config["VGG_FILE"]
            self.vgg_param_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
            print("VGG parameter loaded")

        self.train_file = self.config["TRAIN_FILE"]
        self.val_file = self.config["VAL_FILE"]
        self.test_file = self.config["TEST_FILE"]
        self.img_prefix = self.config["IMG_PREFIX"]
        self.label_prefix = self.config["LABEL_PREFIX"]
        self.bayes = self.config["BAYES"]
        self.opt = self.config["OPT"]
        self.saved_dir = self.config["SAVE_MODEL_DIR"]
        self.input_w = self.config["INPUT_WIDTH"]
        self.input_h = self.config["INPUT_HEIGHT"]
        self.input_c = self.config["INPUT_CHANNELS"]
        self.tb_logs = self.config["TB_LOGS"]
        self.batch_size = self.config["BATCH_SIZE"]

        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_acc = [], []

        self.model_version = 0  # used for saving the model
        self.saver = None
        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.compat.v1.Session()
            self.batch_size_pl = tf.compat.v1.placeholder(tf.int64, shape=[], name="batch_size")
            self.is_training_pl = tf.compat.v1.placeholder(tf.bool, name="is_training")
            self.with_dropout_pl = tf.compat.v1.placeholder(tf.bool, name="with_dropout")
            self.keep_prob_pl = tf.compat.v1.placeholder(tf.float32, shape=None, name="keep_rate")
            self.inputs_pl = tf.compat.v1.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_c])
            self.labels_pl = tf.compat.v1.placeholder(tf.int64, [None, self.input_h, self.input_w, 1])

            # Before enter the images into the architecture, we need to do Local Contrast Normalization
            # But it seems a bit complicated, so we use Local Response Normalization which implement in Tensorflow
            # Reference page:https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
            self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
            # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
            self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

            # Second box of convolution layer(4)
            self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

            # Third box of convolution layer(7)
            self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

            # Fourth box of convolution layer(10)
            if self.bayes:
                self.dropout1 = tf.compat.v1.layers.dropout(self.pool3, rate=(1 - self.keep_prob_pl),
                                                            training=self.with_dropout_pl, name="dropout1")
                self.conv4_1 = conv_layer(self.dropout1, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            else:
                self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

            # Fifth box of convolution layers(13)
            if self.bayes:
                self.dropout2 = tf.compat.v1.layers.dropout(self.pool4, rate=(1 - self.keep_prob_pl),
                                                            training=self.with_dropout_pl, name="dropout2")
                self.conv5_1 = conv_layer(self.dropout2, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            else:
                self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                          self.vgg_param_dict)
            self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.is_training_pl, self.use_vgg,
                                      self.vgg_param_dict)
            self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

            # ---------------------So Now the encoder process has been Finished--------------------------------------#
            # ------------------Then Let's start Decoder Process-----------------------------------------------------#

            # First box of deconvolution layers(3)
            if self.bayes:
                self.dropout3 = tf.compat.v1.layers.dropout(self.pool5, rate=(1 - self.keep_prob_pl),
                                                            training=self.with_dropout_pl, name="dropout3")
                self.deconv5_1 = up_sampling(self.dropout3, self.pool5_index, self.shape_5, self.batch_size_pl,
                                             name="unpool_5")
            else:
                self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, self.batch_size_pl,
                                             name="unpool_5")
            self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512], self.is_training_pl)
            self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512], self.is_training_pl)
            self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512], self.is_training_pl)
            # Second box of deconvolution layers(6)
            if self.bayes:
                self.dropout4 = tf.compat.v1.layers.dropout(self.deconv5_4, rate=(1 - self.keep_prob_pl),
                                                            training=self.with_dropout_pl, name="dropout4")
                self.deconv4_1 = up_sampling(self.dropout4, self.pool4_index, self.shape_4, self.batch_size_pl,
                                             name="unpool_4")
            else:
                self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, self.batch_size_pl,
                                             name="unpool_4")
            self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512], self.is_training_pl)
            self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512], self.is_training_pl)
            self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256], self.is_training_pl)
            # Third box of deconvolution layers(9)
            if self.bayes:
                self.dropout5 = tf.compat.v1.layers.dropout(self.deconv4_4, rate=(1 - self.keep_prob_pl),
                                                            training=self.with_dropout_pl, name="dropout5")
                self.deconv3_1 = up_sampling(self.dropout5, self.pool3_index, self.shape_3, self.batch_size_pl,
                                             name="unpool_3")
            else:
                self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, self.batch_size_pl,
                                             name="unpool_3")
            self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256], self.is_training_pl)
            self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256], self.is_training_pl)
            self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128], self.is_training_pl)
            # Fourth box of deconvolution layers(11)
            if self.bayes:
                self.dropout6 = tf.compat.v1.layers.dropout(self.deconv3_4, rate=(1 - self.keep_prob_pl),
                                                            training=self.with_dropout_pl, name="dropout6")
                self.deconv2_1 = up_sampling(self.dropout6, self.pool2_index, self.shape_2, self.batch_size_pl,
                                             name="unpool_2")
            else:
                self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, self.batch_size_pl,
                                             name="unpool_2")
            self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128], self.is_training_pl)
            self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64], self.is_training_pl)
            # Fifth box of deconvolution layers(13)
            self.deconv1_1 = up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, self.batch_size_pl,
                                         name="unpool_1")
            self.deconv1_2 = conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64], self.is_training_pl)
            self.deconv1_3 = conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64], self.is_training_pl)

            with tf.compat.v1.variable_scope('conv_classifier') as scope:
                self.kernel = variable_with_weight_decay('weights', initializer=initialization(1, 64),
                                                         shape=[1, 1, 64, self.num_classes], wd=False)
                self.conv = tf.nn.conv2d(input=self.deconv1_3, filters=self.kernel, strides=[1, 1, 1, 1],
                                         padding='SAME')
                self.biases = variable_with_weight_decay('biases', tf.compat.v1.constant_initializer(0.0),
                                                         shape=[self.num_classes], wd=False)
                self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)

    def restore(self):
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        index = 0
        candidate_res_file = 'model.ckpt-0'
        res_file_dir = self.saved_dir
        files_in_dir = os.listdir(res_file_dir)
        if not files_in_dir:
            return
        res_file = None
        while any(f.startswith(f'{candidate_res_file}.') for f in files_in_dir):
            res_file = candidate_res_file
            index += 1
            candidate_res_file = f'model.ckpt-{index}'
        res_file = os.path.join(res_file_dir, res_file)
        print_tensors_in_checkpoint_file(res_file, None, False)
        with self.graph.as_default():
            if self.saver is None:
                self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
            self.saver.restore(self.sess, res_file)
            self.sess = tf.compat.v1.Session()
            self.model_version = index

    def train(self, max_steps=30001, batch_size=3, train_duration=None, validate_duration=None):
        # For train the bayes, the FLAG_OPT SHOULD BE SGD, BUT FOR TRAIN THE NORMAL SEGNET,
        # THE FLAG_OPT SHOULD BE ADAM!!!

        # image_filename, label_filename = get_filename_list(self.train_file, self.config)
        # val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)
        train_start = datetime.now()
        train_data_generator, test_data_generator = get_dataset_generator(batch_size=batch_size, skip_frames=0)
        train_data = train_data_generator()

        with self.graph.as_default():
            # train_dataset = Dataset.from_generator(train_data, output_types=tf.int32).make_one_shot_iterator()
            # test_dataset = Dataset.from_generator(test_data, output_types=tf.int32).make_one_shot_iterator()
            # if self.images_tr is None:
            #     self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, batch_size, self.config)
            #     self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size,
            #                                                       self.config)

            loss, accuracy, prediction = cal_loss(logits=self.logits, labels=self.labels_pl)
            train, global_step = train_op(total_loss=loss, opt=self.opt)

            summary_op = tf.compat.v1.summary.merge_all()

            # steps_wo_improvement = 0

            with self.sess.as_default():
                self.sess.run(tf.compat.v1.local_variables_initializer())
                self.sess.run(tf.compat.v1.global_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.compat.v1.train.start_queue_runners(coord=coord)
                # The queue runners basic reference:
                # https://www.tensorflow.org/versions/r0.12/how_tos/threading_and_queues
                train_writer = tf.compat.v1.summary.FileWriter(self.tb_logs, self.sess.graph)
                for step in range(max_steps):
                    # image_batch, label_batch = self.sess.run([self.images_tr, self.labels_tr])
                    try:
                        image_batch, label_batch = next(train_data)
                    except StopIteration:
                        break
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: True,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: batch_size}

                    _, _loss, _accuracy, summary = self.sess.run([train, loss, accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                    # if self.train_loss and _loss < min(self.train_loss):
                    #     steps_wo_improvement = 0
                    # else:
                    #     steps_wo_improvement += 1
                    self.train_loss.append(_loss)
                    self.train_accuracy.append(_accuracy)

                    if step % 100 == 0:
                        # print()
                        # conv_classifier = self.sess.run(self.logits, feed_dict=feed_dict)
                        # print('per_class accuracy by logits in training time',
                        #       per_class_acc(conv_classifier, label_batch, self.num_classes))
                        # per_class_acc is a function from utils
                        train_writer.add_summary(summary, step)

                    elapsed_time = datetime.now() - train_start
                    elapsed_hours = elapsed_time.total_seconds() / 3600
                    print(f"\rIteration {step}: Train Loss {self.train_loss[-1]:6.3f}, "
                          f"Train Acc {self.train_accuracy[-1]:6.3f}, Elapsed {elapsed_hours:.2f} hr",
                          end='', flush=True)

                    if step != 0 and step % 2000 == 0:
                        self.save()
                    if train_duration is not None:

                        if elapsed_time > train_duration:
                            break
                    # elif steps_wo_improvement == 20:
                    #     break

                print()
                self.save()
                print("\nstart validating.......")
                _val_loss = []
                _val_acc = []
                hist = np.zeros((self.num_classes, self.num_classes))

                test_data = get_full_data_generator(batch_size=batch_size, simple_samples=100, complex_samples=100)

                validation_start = datetime.now()

                with open('per_frame_data.csv', 'w') as f:
                    csv_writer = csv.DictWriter(f, fieldnames=["Filename",
                                                               "Best match",
                                                               "Worst match",
                                                               "Average match",
                                                               "Total frames",
                                                               "Median",
                                                               "Stddev",
                                                               "Sex",
                                                               'Average mIoU'])
                    csv_writer.writeheader()
                    for file_index, (filename, file_frame_generator) in enumerate(test_data):
                        elapsed_time = datetime.now() - validation_start
                        if validate_duration and elapsed_time > validate_duration:
                            break
                        per_file_miou = []
                        per_file_csi = []

                        for index, (image_batch_val, label_batch_val) in enumerate(file_frame_generator):
                            if index % 5 != 0:
                                continue
                            elapsed_time = datetime.now() - validation_start
                            hours = elapsed_time.total_seconds() / 3600
                            if validate_duration and elapsed_time > validate_duration:
                                break
                            print(f'\rValidating file {os.path.basename(filename)}, batch {index}, time taken {hours:.2f} hr', end='',
                                  flush=True)
                            fetches_valid = [loss, accuracy, self.logits]
                            feed_dict_valid = {self.inputs_pl: image_batch_val,
                                               self.labels_pl: label_batch_val,
                                               self.is_training_pl: True,
                                               self.keep_prob_pl: 1.0,
                                               self.with_dropout_pl: False,
                                               self.batch_size_pl: batch_size}
                            # since we still using mini-batch, so in the batch norm we set phase_train to be
                            # true, and because we didin't run the trainop process, so it will not update
                            # the weight!
                            _loss, _acc, _val_pred = self.sess.run(fetches_valid, feed_dict_valid)
                            _val_loss.append(_loss)
                            # if index % 100 == 0:
                            # for i in range(_val_pred.shape[0]):
                            #     img = _val_pred[i]
                            #     self._save_img(i, img, index, 'test_output')
                            # for i in range(label_batch_val.shape[0]):
                            #     img = label_batch_val[i]
                            #     one_hot_img = np.eye(13)[img.reshape([img.shape[0], img.shape[1]])]
                            #     self._save_img(i, one_hot_img, index, 'label')

                            for i in range(_val_pred.shape[0]):
                                expected = label_batch_val[i].reshape(label_batch_val[i].shape[:2])
                                actual = predicted_value_to_image(_val_pred[i]).reshape(_val_pred[i].shape[:2])
                                miou = get_mIoU(expected, actual)
                                csi = get_csi(expected, actual)
                                per_file_miou.append(miou)
                                per_file_csi.append(csi)

                            _val_acc.append(_acc)
                            hist += get_hist(_val_pred, label_batch_val)
                        if per_file_csi:
                            avg = sum(per_file_csi) / len(per_file_csi)
                            csv_writer.writerow({
                                'Filename': os.path.basename(filename),
                                'Best match': max(per_file_csi),
                                "Worst match": min(per_file_csi),
                                "Average match": avg,
                                "Total frames": len(per_file_csi),
                                "Median": sorted(per_file_csi)[len(per_file_csi) // 2],
                                "Stddev": sum((i - avg) ** 2 for i in per_file_csi) / len(per_file_csi),
                                "Average mIoU": sum(per_file_miou) / len(per_file_miou),
                                "Sex": 'M' if is_male_file(filename) else 'F'
                            })

                print_hist_summary(hist)

                self.val_loss.append(np.mean(_val_loss))
                self.val_acc.append(np.mean(_val_acc))
                coord.request_stop()
                coord.join(threads)

    def _save_img(self, i, img, index, prefix):
        raw_data = predicted_value_to_image(img)
        image_data = (raw_data * 255).astype('uint16')
        imsave(os.path.join('segnet_images', f"{prefix}_{index}_{i}.png"), image_data, check_contrast=False)

    def visual_results(self, dataset_type="TEST", images_index=3, FLAG_MAX_VOTE=False):

        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:

            # Restore saved session
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, train_dir)

            _, _, prediction = cal_loss(logits=self.logits,
                                        labels=self.labels_pl)
            prob = tf.nn.softmax(self.logits, axis=-1)

            if (dataset_type == 'TRAIN'):
                test_type_path = self.config["TRAIN_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(367), images_index)
                # indexes = [0,75,150,225,300]
            elif (dataset_type == 'VAL'):
                test_type_path = self.config["VAL_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(101), images_index)
                # indexes = [0,25,50,75,100]
            elif (dataset_type == 'TEST'):
                test_type_path = self.config["TEST_FILE"]
                if type(images_index) == list:
                    indexes = images_index
                else:
                    indexes = random.sample(range(233), images_index)
                # indexes = [0,50,100,150,200]

            # Load images
            image_filename, label_filename = get_filename_list(test_type_path, self.config)
            images, labels = get_all_test_data(image_filename, label_filename)

            # Keep images subset of length images_index
            images = [images[i] for i in indexes]
            labels = [labels[i] for i in indexes]

            num_sample_generate = 30
            pred_tot = []
            var_tot = []

            for image_batch, label_batch in zip(images, labels):

                image_batch = np.reshape(image_batch, [1, image_h, image_w, image_c])
                label_batch = np.reshape(label_batch, [1, image_h, image_w, 1])

                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: False,
                                 self.keep_prob_pl: 0.5,
                                 self.batch_size_pl: 1}
                    pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                    pred = np.reshape(pred, [image_h, image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: False,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: 1}
                    prob_iter_tot = []
                    pred_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches=[prob], feed_dict=feed_dict)
                        prob_iter_tot.append(prob_iter_step)
                        pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step, axis=-1), [-1]))

                    if FLAG_MAX_VOTE is True:
                        prob_variance, pred = MAX_VOTE(pred_iter_tot, prob_iter_tot, self.config["NUM_CLASSES"])
                        # acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                        var_one = var_calculate(pred, prob_variance)
                        pred = np.reshape(pred, [image_h, image_w])
                    else:
                        prob_mean = np.nanmean(prob_iter_tot, axis=0)
                        prob_variance = np.var(prob_iter_tot, axis=0)
                        pred = np.reshape(np.argmax(prob_mean, axis=-1),
                                          [-1])  # pred is the predicted label with the mean of generated samples
                        # THIS TIME I DIDN'T INCLUDE TAU
                        var_one = var_calculate(pred, prob_variance)
                        pred = np.reshape(pred, [image_h, image_w])

                pred_tot.append(pred)
                var_tot.append(var_one)

            draw_plots_bayes(images, labels, pred_tot, var_tot)

    def visual_results_external_image(self, images, FLAG_MAX_VOTE=False):

        # train_dir = "./saved_models/segnet_vgg_bayes/segnet_vgg_bayes_30000/model.ckpt-30000"
        # train_dir = "./saved_models/segnet_scratch/segnet_scratch_30000/model.ckpt-30000"

        i_width = 480
        i_height = 360
        images = [misc.imresize(image, (i_height, i_width)) for image in images]

        image_w = self.config["INPUT_WIDTH"]
        image_h = self.config["INPUT_HEIGHT"]
        image_c = self.config["INPUT_CHANNELS"]
        train_dir = self.config["SAVE_MODEL_DIR"]
        FLAG_BAYES = self.config["BAYES"]

        with self.sess as sess:

            # Restore saved session
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, train_dir)

            _, _, prediction = cal_loss(logits=self.logits,
                                        labels=self.labels_pl)
            prob = tf.nn.softmax(self.logits, axis=-1)

            num_sample_generate = 30
            pred_tot = []
            var_tot = []

            labels = []
            for i in range(len(images)):
                labels.append(np.array([[1 for x in range(480)] for y in range(360)]))

            inference_time = []
            start_time = time.time()

            for image_batch, label_batch in zip(images, labels):
                # for image_batch in zip(images):

                image_batch = np.reshape(image_batch, [1, image_h, image_w, image_c])
                label_batch = np.reshape(label_batch, [1, image_h, image_w, 1])

                if FLAG_BAYES is False:
                    fetches = [prediction]
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: False,
                                 self.keep_prob_pl: 0.5,
                                 self.batch_size_pl: 1}
                    pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                    pred = np.reshape(pred, [image_h, image_w])
                    var_one = []
                else:
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 self.is_training_pl: False,
                                 self.keep_prob_pl: 0.5,
                                 self.with_dropout_pl: True,
                                 self.batch_size_pl: 1}
                    prob_iter_tot = []
                    pred_iter_tot = []
                    for iter_step in range(num_sample_generate):
                        prob_iter_step = sess.run(fetches=[prob], feed_dict=feed_dict)
                        prob_iter_tot.append(prob_iter_step)
                        pred_iter_tot.append(np.reshape(np.argmax(prob_iter_step, axis=-1), [-1]))

                    if FLAG_MAX_VOTE is True:
                        prob_variance, pred = MAX_VOTE(pred_iter_tot, prob_iter_tot, self.config["NUM_CLASSES"])
                        # acc_per = np.mean(np.equal(pred,np.reshape(label_batch,[-1])))
                        var_one = var_calculate(pred, prob_variance)
                        pred = np.reshape(pred, [image_h, image_w])
                    else:
                        prob_mean = np.nanmean(prob_iter_tot, axis=0)
                        prob_variance = np.var(prob_iter_tot, axis=0)
                        pred = np.reshape(np.argmax(prob_mean, axis=-1),
                                          [-1])  # pred is the predicted label with the mean of generated samples
                        # THIS TIME I DIDN'T INCLUDE TAU
                        var_one = var_calculate(pred, prob_variance)
                        pred = np.reshape(pred, [image_h, image_w])

                pred_tot.append(pred)
                var_tot.append(var_one)
                inference_time.append(time.time() - start_time)
                start_time = time.time()

            try:
                draw_plots_bayes_external(images, pred_tot, var_tot)
                return pred_tot, var_tot, inference_time
            except:
                return pred_tot, var_tot, inference_time

    def test(self):
        image_filename, label_filename = get_filename_list(self.test_file, self.config)

        with self.graph.as_default():
            with self.sess as sess:
                loss, accuracy, prediction = normal_loss(self.logits, self.labels_pl, self.num_classes)
                prob = tf.nn.softmax(self.logits, axis=-1)
                prob = tf.reshape(prob, [self.input_h, self.input_w, self.num_classes])

                images, labels = get_all_test_data(image_filename, label_filename)

                NUM_SAMPLE = []
                for i in range(30):
                    NUM_SAMPLE.append(2 * i + 1)

                acc_final = []
                iu_final = []
                iu_mean_final = []
                # uncomment the line below to only run for two times.
                # NUM_SAMPLE = [1, 30]
                NUM_SAMPLE = [1]
                for num_sample_generate in NUM_SAMPLE:

                    loss_tot = []
                    acc_tot = []
                    pred_tot = []
                    var_tot = []
                    hist = np.zeros((self.num_classes, self.num_classes))
                    step = 0
                    for image_batch, label_batch in zip(images, labels):
                        image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                        label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
                        # comment the code below to apply the dropout for all the samples
                        if num_sample_generate == 1:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: False,
                                         self.batch_size_pl: 1}
                        else:
                            feed_dict = {self.inputs_pl: image_batch, self.labels_pl: label_batch,
                                         self.is_training_pl: False,
                                         self.keep_prob_pl: 0.5, self.with_dropout_pl: True,
                                         self.batch_size_pl: 1}
                        # uncomment this code below to run the dropout for all the samples
                        # feed_dict = {test_data_tensor: image_batch, test_label_tensor:label_batch, phase_train: False, keep_prob:0.5, phase_train_dropout:True}
                        fetches = [loss, accuracy, self.logits, prediction]
                        if self.bayes is False:
                            loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                            var_one = []
                        else:
                            logit_iter_tot = []
                            loss_iter_tot = []
                            acc_iter_tot = []
                            prob_iter_tot = []
                            logit_iter_temp = []
                            for iter_step in range(num_sample_generate):
                                loss_iter_step, acc_iter_step, logit_iter_step, prob_iter_step = sess.run(
                                    fetches=[loss, accuracy, self.logits, prob], feed_dict=feed_dict)
                                loss_iter_tot.append(loss_iter_step)
                                acc_iter_tot.append(acc_iter_step)
                                logit_iter_tot.append(logit_iter_step)
                                prob_iter_tot.append(prob_iter_step)
                                logit_iter_temp.append(
                                    np.reshape(logit_iter_step, [self.input_h, self.input_w, self.num_classes]))

                            loss_per = np.nanmean(loss_iter_tot)
                            acc_per = np.nanmean(acc_iter_tot)
                            logit = np.nanmean(logit_iter_tot, axis=0)
                            print(np.shape(prob_iter_tot))

                            prob_mean = np.nanmean(prob_iter_tot, axis=0)
                            prob_variance = np.var(prob_iter_tot, axis=0)
                            logit_variance = np.var(logit_iter_temp, axis=0)

                            # THIS TIME I DIDN'T INCLUDE TAU
                            pred = np.reshape(np.argmax(prob_mean, axis=-1), [-1])  # pred is the predicted label

                            var_sep = []  # var_sep is the corresponding variance if this pixel choose label k
                            length_cur = 0  # length_cur represent how many pixels has been read for one images
                            for row in np.reshape(prob_variance, [self.input_h * self.input_w, self.num_classes]):
                                temp = row[pred[length_cur]]
                                length_cur += 1
                                var_sep.append(temp)
                            var_one = np.reshape(var_sep, [self.input_h,
                                                           self.input_w])  # var_one is the corresponding variance in terms of the "optimal" label
                            pred = np.reshape(pred, [self.input_h, self.input_w])

                        loss_tot.append(loss_per)
                        acc_tot.append(acc_per)
                        pred_tot.append(pred)
                        var_tot.append(var_one)
                        print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1],
                                                                                           acc_tot[-1]))
                        step = step + 1
                        per_class_acc(logit, label_batch, self.num_classes)
                        hist += get_hist(logit, label_batch)

                    acc_tot = np.diag(hist).sum() / hist.sum()
                    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

                    print("Total Accuracy for test image: ", acc_tot)
                    print("Total MoI for test images: ", iu)
                    print("mean MoI for test images: ", np.nanmean(iu))

                    acc_final.append(acc_tot)
                    iu_final.append(iu)
                    iu_mean_final.append(np.nanmean(iu))

            return acc_final, iu_final, iu_mean_final, prob_variance, logit_variance, pred_tot, var_tot

    def save(self):
        os.makedirs(os.path.join(self.saved_dir, "Data"), exist_ok=True)
        np.save(os.path.join(self.saved_dir, "Data", "trainloss"), self.train_loss)
        np.save(os.path.join(self.saved_dir, "Data", "trainacc"), self.train_accuracy)
        np.save(os.path.join(self.saved_dir, "Data", "valloss"), self.val_loss)
        np.save(os.path.join(self.saved_dir, "Data", "valacc"), self.val_acc)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver = tf.compat.v1.train.Saver()
                checkpoint_path = os.path.join(self.saved_dir, 'model.ckpt')
                path = self.saver.save(self.sess, checkpoint_path, global_step=self.model_version)
                print(f'Saved model to {path}')
                self.model_version += 1


def predicted_value_to_image(predicted):
    return (np.argmax(predicted, axis=2) == 1).astype(int)


def get_mIoU(img1: np.ndarray, img2: np.ndarray) -> float:
    i = get_intersection(img1, img2)
    u = get_union(img1, img2)
    if u == 0:
        return 0.0
    return i / u


def get_csi(img1: np.ndarray, img2: np.ndarray) -> float:
    i = get_intersection(img1, img2)
    img1_positives = np.count_nonzero(img1 == 1)
    img2_positives = np.count_nonzero(img2 == 1)
    if img1_positives == 0 or img2_positives == 0:
        return 0.0
    return (i / img1_positives) * (i / img2_positives)


def get_intersection(img1: np.ndarray, img2: np.ndarray) -> float:
    return _get_logical_op_result(img1, img2, np.logical_and)


def get_union(img1, img2) -> float:
    return _get_logical_op_result(img1, img2, np.logical_or)


def _get_logical_op_result(img1: np.ndarray, img2: np.ndarray, operation) -> float:
    img1_bool = img1.astype('bool')
    img2_bool = img2.astype('bool')
    product = operation(img1_bool, img2_bool)
    return np.count_nonzero(product == True)
