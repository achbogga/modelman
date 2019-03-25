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

import pycuda.driver as cuda
# This import causes pycuda to automatically
# manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys
import os


def rt_infer(uff_model_file):
    # Build a TensorRT engine given an uff model file path and perform inference
    with build_engine_uff(uff_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            test_image = random.choice(test_images)
            test_case = load_normalized_test_case(test_image, h_input)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            pred = labels[np.argmax(h_output)]
            if "_".join(pred.split()) in os.path.splitext(os.path.basename(test_case))[0]:
                print("Correctly recognized " + test_case + " as " + pred)
            else:
                print("Incorrectly recognized " + test_case + " as " + pred)
