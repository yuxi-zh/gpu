import json
import numpy         as np
import pycuda.driver as drv
import os, sys
from neon.initializers import Gaussian
from neon.layers import Convolution
from neon.backends import gen_backend

N, C, K, \
D, H, W, \
T, R, S, \
pad_d, pad_h, pad_w, \
str_d, str_h, str_w, \
dil_d, dil_h, dil_w = [int(v) for v in sys.argv[1:]]

be = gen_backend('gpu', batch_size=N)
conv_layer = Convolution((1, R, S, K),
    strides={'str_d':str_d, 'str_h':str_h, 'str_w':str_w},
    padding={'pad_d':pad_d, 'pad_h':pad_h, 'pad_w':pad_w},
    dilation={'dil_d':dil_d, 'dil_h':dil_h, 'dil_w':dil_w})
conv_layer.configure((C, 1, H, W))
fprop_kernels = conv_layer.nglayer.fprop_kernels
with open('KernelQueryResult', 'w') as of:
	json.dump({
		"kernel_name":fprop_kernels.kernel_name, 
		"kernel_args":fprop_kernels.kernel_args
	}, of);