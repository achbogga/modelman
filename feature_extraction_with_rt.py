"""Imports a frozen model graph uff file.
converts the model into RT engine
and saves the engine file for future inference
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


import argparse
# This sample uses a UFF ResNet50 Model to create a TensorRT Inference Engine
import random
from PIL import Image
import numpy as np

import pycuda.driver as cuda
# This import causes pycuda to automatically
# manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))


class ModelData(object):
    MODEL_PATH = "resnet50-infer-5.uff"
    INPUT_NAME = "input"
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_NAME = "GPU_0/tower_0/Softmax"
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress
# messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_workspace_size_in_gigs(val):
    return val * 1 << 30


def allocate_buffers(engine):
    # Allocate host and device buffers, and create a stream.
    # Determine dimensions and create page-locked memory buffers
    # (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),
                                    dtype=trt.nptype(ModelData.DTYPE))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),
                                     dtype=trt.nptype(ModelData.DTYPE))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)],
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()


def build_engine_uff(model_file):
    # The UFF path is used for TensorFlow models.
    # You can convert a frozen TensorFlow graph to UFF
    # using the included convert-to-uff utility.
    # You can set the logger severity higher to
    # suppress messages (or lower to display more messages).
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        # Workspace size is the maximum amount of memory available to the builder while building an engine.
        # It should generally be set as high as possible.
        builder.max_workspace_size = get_workspace_size_in_gigs(1)
        # We need to manually register the input and output nodes for UFF.
        parser.register_input(ModelData.INPUT_NAME, ModelData.INPUT_SHAPE)
        parser.register_output(ModelData.OUTPUT_NAME)
        # Load the UFF model and parse it in order to populate the TensorRT network.
        parser.parse(model_file, network)
        # Build and return an engine.
        return builder.build_cuda_engine(network)


def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image


def main(args):
    if args.mode == 'infer':
        # Perform inference, save the features and the engine file.
        pass
    else:
        # Just save the engine file.
        pass


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str,
                        help='The infer mode or save_engine mode')

    parser.add_argument('dataset_dir', type=str,
                        help='Directory containing the images to be \
                            to be sent for the inference')
    parser.add_argument('model_path', type=str,
                        help='The path to the frozen uff file.')
    parser.add_argument('output_dir', type=str,
                        help='The output path to write the engine file to.')
    parser.add_argument('--output_node_names', type=str, nargs='+',
                        help='Output node names for the exported\
                            graphdef protobuf (.pb)',
                        default=['embeddings', 'label_batch'])
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
