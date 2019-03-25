"""Imports a model ckpt and extracts features.
Saves them as npy files along with their labels.
"""
# MIT License
#
# Copyright (c) 2019 Achyut Sarma Boggaram
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_models_io import get_image_paths, get_model_filenames, load_model
from image_io import *
import argparse
import imageio
import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
import sys
import tensorflow as tf


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str,
                        help='The directory containing the input images')
    parser.add_argument('output_dir', type=str,
                        help='The directory to write the features to')
    parser.add_argument('model_dir', type=str,
                        help='The directory containing the trained model')
    parser.add_argument('--percentage', type=float,
                        help='The percentage of the input to be processed',
                        default=1.0)
    parser.add_argument('--image_size', type=int,
                        help='The percentage of the input to be processed',
                        default=299)
    parser.add_argument('--output_prefix', type=str,
                        help='The output prefix name for the features',
                        default='center_loss_embedding')
    parser.add_argument('--input_tensor_name', type=str,
                        help='The tensor name to feed the input images',
                        default='input:0')
    parser.add_argument('--output_tensor_name', type=str,
                        help='The tensor name to extract the features from',
                        default='embeddings:0')
    parser.add_argument('--phase_train_tensor', action='store_true')
    parser.add_argument('--use_rt', action='store_true')
    return parser.parse_args(argv)


def get_embeddings(image_names, model_dir, chunk_size=500,
                   features_tensor_name="embeddings:0",
                   phase_train_tensor=False, image_size=299,
                   input_tensor_name="input:0"):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model(model_dir)
            images_placeholder = \
                tf.get_default_graph().get_tensor_by_name(input_tensor_name)
            embeddings_placeholder = \
                tf.get_default_graph().get_tensor_by_name(features_tensor_name)
            if phase_train_tensor:
                phase_train_placeholder = \
                    tf.get_default_graph().get_tensor_by_name("phase_train:0")
            if len(image_names) < chunk_size:
                print('Total size: ', len(image_names))
                images = np.asarray([prewhiten(imageio.imread(image_name))
                                     for image_name in image_names])
                images = [zoom(im, [image_size / float(im.shape[0]),
                                    image_size / float(im.shape[1]), 1.0])
                          for im in images]
                if phase_train_tensor:
                    feed_dict = {images_placeholder: images,
                                 phase_train_placeholder: False}
                else:
                    feed_dict = {images_placeholder: images}
                embeddings = sess.run(embeddings_placeholder,
                                      feed_dict=feed_dict)
            else:
                print('Total size: ', len(image_names))
                i = 0
                image_names_chunk = image_names[i:min(
                    i + chunk_size, len(image_names))]
                images_chunk = np.asarray(
                    [prewhiten(imageio.imread(image_name))
                     for image_name in image_names_chunk])
                images_chunk = [zoom(im, [image_size / float(im.shape[0]),
                                          image_size / float(im.shape[1]), 1.0])
                                for im in images_chunk]
                if phase_train_tensor:
                    feed_dict = {images_placeholder: images_chunk,
                                 phase_train_placeholder: False}
                else:
                    feed_dict = {images_placeholder: images_chunk}
                embeddings = sess.run(
                    embeddings_placeholder, feed_dict=feed_dict)
                i += chunk_size
                for i in range(i, len(image_names), chunk_size):
                    print('current chunk i: ', i)
                    image_names_chunk = image_names[i:min(
                        i + chunk_size, len(image_names))]
                    images_chunk = np.asarray(
                        [prewhiten(imageio.imread(image_name))
                         for image_name in image_names_chunk])
                    images_chunk = [zoom(
                        im, [image_size / float(im.shape[0]),
                             image_size / float(im.shape[1]), 1.0])
                        for im in images_chunk]
                    if phase_train_tensor:
                        feed_dict = {images_placeholder: images_chunk,
                                     phase_train_placeholder: False}
                    else:
                        feed_dict = {images_placeholder: images_chunk}
                    embeddings_chunk = sess.run(
                        embeddings_placeholder, feed_dict=feed_dict)
                    embeddings = np.vstack((embeddings, embeddings_chunk))
            return np.asarray(embeddings)


def extract_features(training_image_paths, model_dir, chunk_size=512,
                     output_dir='/home/caffe/achu/\
                                 results_ensemble_1_16_19_10Percent_data',
                     percentage=0.1, output_tensor="embeddings:0",
                     input_tensor_name="input:0",
                     image_size=256, phase_train_tensor=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # dataset_prefix = dataset_train_dir.split('_')[0]
    # dataset_prefix = dataset_prefix.split('/')[-1]
    # print ('Dataset: ' + dataset_prefix)
    print('Model: ' + model_dir)

    training_embeddings = []
    training_predictions = []
    training_sku_labels = [int(float(image_path.split('/')[-2]))
                           for image_path in training_image_paths]
    if percentage < 1.0:
        print('percentage: ', percentage)
        training_image_paths_split_1, training_image_paths_split_2, training_sku_labels_split_1, training_sku_labels_split_2 = train_test_split(
            training_image_paths, training_sku_labels,
            train_size=percentage, random_state=0)
    else:
        training_image_paths_split_1 = training_image_paths
        training_sku_labels_split_1 = training_sku_labels
    sku_set = set(training_sku_labels_split_1)
    training_embeddings = get_embeddings(training_image_paths_split_1,
                                         model_dir, chunk_size=chunk_size,
                                         features_tensor_name=output_tensor,
                                         phase_train_tensor=phase_train_tensor,
                                         input_tensor_name=input_tensor_name,
                                         image_size=image_size)

    training_data = (training_embeddings, training_sku_labels_split_1)
    return training_data


def save_features(output_dir, percentage, dataset_dir, model_dir, prefix,
                  output_tensor_name="embeddings:0",
                  input_tensor_name="input:0",
                  phase_train_tensor=True,
                  image_size=299):
    training_image_paths = get_image_paths(dataset_dir)
    training_data = extract_features(training_image_paths, model_dir,
                                     output_dir=output_dir,
                                     percentage=percentage,
                                     output_tensor=output_tensor_name,
                                     input_tensor_name=input_tensor_name,
                                     phase_train_tensor=phase_train_tensor,
                                     image_size=image_size)
    training_embeddings = training_data[0]
    training_labels = training_data[1]
    np.save(output_dir + '/' + prefix + '_features.npy', training_embeddings)
    np.save(output_dir + '/' + prefix + '_labels.npy', training_labels)


def main(args):
    save_features(args.output_dir, args.percentage, args.input_dir,
                  args.model_dir, args.output_prefix, args.output_tensor_name,
                  input_tensor_name=args.input_tensor_name,
                  phase_train_tensor=args.phase_train_tensor,
                  image_size=args.image_size)
    pass


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
