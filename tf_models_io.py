# Most frequently used tf model IO functions required
# for many CV/ML research tasks at Clobotics
# Author: Achyut Sarma Boggaram
# Copyrights: Clobotics Corporation

import numpy as np
import tensorflow as tf
import os
from tensorflow.python.platform import gfile


def load_model(model, input_map=None):
    # Check if the model is a model directory
    # (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file),
                                           input_map=input_map)
        saver.restore(tf.get_default_session(),
                      os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    # Reads a tensorflow trained model folder and
    # returns the relevant model file paths
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the \
                         model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one \
                         meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def get_image_paths(dataset_dir):
    # Reads all the image paths from dataset_dir
    sku_labels_str = [sku_label for sku_label in os.listdir(dataset_dir)
                      if os.path.isdir(dataset_dir + '/' + sku_label)]
    sku_paths = [dataset_dir + '/' + sku for sku in sku_labels_str]
    image_paths = []
    for sku in sku_paths:
        sku_image_names = [sku + '/' + f for f in os.listdir(sku)]
        image_paths += sku_image_names
    return image_paths
