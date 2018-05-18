

#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
from __future__ import division
from __future__ import print_function
import numpy         as np
import pycuda.driver as drv
import os, sys
import itertools
from neon import logger as neon_logger
from neon.initializers import Gaussian
from neon.layers import Convolution
from neon.backends import gen_backend
from neon.backends.nervanagpu import NervanaGPU, GPUTensor
from neon.backends.nervanacpu import NervanaCPU

from neon.backends.convolution import (_ceil_div,
    FpropCuda,   BpropCuda,   UpdateCuda,
    FpropDirect, BpropDirect, UpdateDirect)

from neon.backends.winograd_conv import (
    FpropWinograd_2x2_3x3, BpropWinograd_2x2_3x3, UpdateWinograd_3x3_2x2,
    FpropWinograd_4x4_3x3, BpropWinograd_4x4_3x3, UpdateWinograd_3x3_4x4,
    FpropWinograd_2x2_5x5, BpropWinograd_2x2_5x5)

fprop_kernels  = (FpropCuda,  FpropDirect,  FpropWinograd_2x2_3x3,  FpropWinograd_4x4_3x3, FpropWinograd_2x2_5x5)
bprop_kernels  = (BpropCuda,  BpropDirect,  BpropWinograd_2x2_3x3,  BpropWinograd_4x4_3x3, BpropWinograd_2x2_5x5)
update_kernels = (UpdateCuda, UpdateDirect, UpdateWinograd_3x3_2x2, UpdateWinograd_3x3_4x4)


#                D,   H,   W,  T, R, S,    pad,   str
conv_1x1     = ( 1,  14,  14,  1, 1, 1,  0,0,0, 1,1,1)
conv_3x3     = ( 1,  14,  14,  1, 3, 3,  0,1,1, 1,1,1)
conv_3x3p0   = ( 1,  14,  14,  1, 3, 3,  0,0,0, 1,1,1)
conv_3x3p2   = ( 1,  14,  14,  1, 3, 3,  0,2,2, 1,1,1)
conv_3x3s2   = ( 1,  14,  14,  1, 3, 3,  0,1,1, 1,2,2)
conv_1x3     = ( 1,  14,  14,  1, 1, 3,  0,0,1, 1,1,1)
conv_3x1     = ( 1,  14,  14,  1, 3, 1,  0,1,0, 1,1,1)
conv_5x5     = ( 1,  14,  14,  1, 5, 5,  0,2,2, 1,1,1)
conv_11x11s4 = ( 1, 224, 224,  1,11,11,  0,2,2, 1,4,4)
conv_1x1x1   = ( 7,   7,   7,  1, 1, 1,  0,0,0, 1,1,1)
conv_3x3x3   = ( 7,   7,   7,  3, 3, 3,  1,1,1, 1,1,1)
conv_3x3x3s2 = ( 7,   7,   7,  3, 3, 3,  1,1,1, 2,2,2)
conv_3x3L    = ( 1, 200, 200,  1, 3, 3,  0,1,1, 1,1,1)
conv_1D      = ( 1, 13, 3263,  1,13,11,  0,0,0, 1,1,3)

N = [1, 16, 32]

CKDHWTRSPS = [
    [ 3, 64,        1, 224, 224,         1, 7, 7,        0, 3, 3,          1, 2, 2,],
    [ 64, 64,       1, 56, 56,       1, 3, 3,        0, 1, 1,           1, 1, 1,],
    [ 64, 64,       1, 56, 56,       1, 1, 1,        0, 0, 0,           1, 1, 1,],
    [ 64, 128,      1, 56, 56,       1, 3, 3,        0, 1, 1,          1, 2, 2,],
    [ 64, 128,      1, 56, 56,       1, 1, 1,        0, 0, 0,          1, 2, 2,],
    [ 128, 128,         1, 28, 28,       1, 3, 3,        0, 1, 1,         1, 1, 1,],
    [ 128, 256,         1, 28, 28,       1, 3, 3,        0, 1, 1,         1, 2, 2,],
    [ 128, 256,         1, 28, 28,       1, 1, 1,        0, 0, 0,         1, 2, 2,],
    [ 256, 256,         1, 14, 14,       1, 3, 3,        0, 1, 1,         1, 1, 1,],
    [ 256, 512,         1, 14, 14,       1, 3, 3,        0, 1, 1,         1, 2, 2,],
    [ 256, 512,         1, 14, 14,       1, 1, 1,        0, 0, 0,         1, 2, 2,],
    [ 512, 512,         1, 7, 7,        1, 3, 3,         0, 1, 1,           1, 1, 1,],
    [ 128, 128,         1, 122, 122,         1, 3, 3,        0, 1, 1,           1, 1, 1,],
    [ 1, 64,        1, 224, 224,         1, 5, 5,        0, 2, 2,          1, 1, 1,],
    [ 64, 64,       1, 224, 224,         1, 3, 3,        0, 1, 1,         1, 1, 1,],
    [ 64, 32,       1, 224, 224,         1, 3, 3,        0, 1, 1,         1, 1, 1,],
    [ 32, 9,        1, 224, 224,         1, 3, 3,        0, 1, 1,          1, 1, 1,]
]

configs = itertools.product(CKDHWTRSPS, N)

def timing_conv_layer(config):

    other, N = config
    C, K, D, H, W, T, R, S, pad_d, pad_h, pad_w, str_d, str_h, str_w = other
    be = gen_backend('gpu', batch_size=N)
    conv_layer = Convolution((T, R, S, K), 
        strides={'str_d':str_d, 'str_h':str_h, 'str_w':str_w},
        padding={'pad_d':pad_d, 'pad_h':pad_h, 'pad_w':pad_w},
        init=Gaussian())

    conv_layer.configure((C, D, H, W))
    conv_layer.allocate()

    I = GPUTensor(be, (C, D, H, W, N))

    # warmup
    conv_layer.fprop(I, inference=True)

    start = drv.Event()
    end = drv.Event()

    time = 0
    for _ in range(10):
        start.record()
        conv_layer.fprop(I, inference=True)
        end.record()
        end.synchronize()
        time += start.time_till(end)

    conv_layer.nglayer.fprop_kernels

    print(time / 10)

def test_sconv_direct_fprop_128x128():

    N, C, K = 128, 4, 128
    D, H, W = 1, 32, 32
    T, R, S = 1, 3, 3
    pad_d, pad_h, pad_w = 0, 0, 0
    str_d, str_h, str_w = 1, 1, 1
    dil_d, dil_h, dil_w = 1, 1, 1

    be = gen_backend('gpu', batch_size=N)
    conv_layer = Convolution((T, R, S, K),
        strides={'str_d':str_d, 'str_h':str_h, 'str_w':str_w},
        padding={'pad_d':pad_d, 'pad_h':pad_h, 'pad_w':pad_w},
        init=Gaussian())

    conv_layer.configure((C, D, H, W))
    print(conv_layer.in_shape, conv_layer.out_shape)
    print(conv_layer.nglayer.dimI, conv_layer.nglayer.dimO)
    print(type(conv_layer.nglayer.fprop_kernels))
    print(conv_layer.nglayer.fprop_kernels.kernel_args)
    print(conv_layer.nglayer.fprop_kernels.kernel_name)

    conv_layer.allocate()
    I = GPUTensor(be, (C, D, H, W, N))
    O = conv_layer.fprop(I, inference=True)

#for config in configs:
#    print(config, end=' ')
#    timing_conv_layer(config)

test_sconv_direct_fprop_128x128()
