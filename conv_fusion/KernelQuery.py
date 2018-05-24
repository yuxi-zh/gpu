import json
import numpy         as np
import pycuda.driver as drv
import os, sys
from neon.initializers import Gaussian
from neon.layers import Convolution
from neon.backends import gen_backend

kernels = {
    "sconv_direct_fprop_128x128": {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_128x128": {"threads": 256, "sass": "sconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_128x128": {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "fprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_128x128": {"threads": 256, "sass": "hconv_xprop_X128_N128", "params": "bprop",  "share": "128*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_64x128":  {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_64x128":  {"threads": 128, "sass": "sconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_64x128":  {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "fprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_64x128":  {"threads": 128, "sass": "hconv_xprop_X64_N128",  "params": "bprop",  "share": " 64*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_32x128":  {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_32x128":  {"threads":  64, "sass": "sconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_32x128":  {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "fprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_32x128":  {"threads":  64, "sass": "hconv_xprop_X32_N128",  "params": "bprop",  "share": " 32*8*2 + 128*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_128x64":  {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_128x64":  {"threads": 128, "sass": "sconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_128x64":  {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "fprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_128x64":  {"threads": 128, "sass": "hconv_xprop_X128_N64",  "params": "bprop",  "share": "128*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_64x64":   {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "sconv_direct_bprop_64x64":   {"threads":  64, "sass": "sconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},
    "hconv_direct_fprop_64x64":   {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "fprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "f"}},
    "hconv_direct_bprop_64x64":   {"threads":  64, "sass": "hconv_xprop_X64_N64",   "params": "bprop",  "share": " 64*8*2 +  64*8*2 + 10", "args": {"prop": "b"}},

    "sconv_direct_fprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "fprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"s","prop":"f"}},
    "sconv_direct_bprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "bprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"s","prop":"b"}},
    "hconv_direct_fprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "fprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"h","prop":"f"}},
    "hconv_direct_bprop_64x32":   {"threads": 128, "sass": "xconv_direct_xprop_64x32",  "params": "bprop2", "share": "(32 + 64)*32*2 + 4", "args": {"type":"h","prop":"b"}},
    "sconv_direct_updat_64x32":   {"threads": 128, "sass": "xconv_direct_updat_64x32",  "params": "updat2", "share": "(32 + 64)*33*2 + 8", "args": {"type": "s",}},
    "hconv_direct_updat_64x32":   {"threads": 128, "sass": "xconv_direct_updat_64x32",  "params": "updat2", "share": "(32 + 64)*33*2 + 8", "args": {"type": "h",}},


    "sconv_winograd_2x2_3x3_32x32":   {"threads": 256, "sass": "xconv_winograd_2x2_3x3_32x32",   "params": "fpropw", "share": "512*4*4", "args": {"type": "s"}},
    "hconv_winograd_2x2_3x3_32x32":   {"threads": 256, "sass": "xconv_winograd_2x2_3x3_32x32",   "params": "fpropw", "share": "512*4*4", "args": {"type": "h"}},
    "sconv_winograd_3x3_2x2_32x32":   {"threads": 256, "sass": "xconv_winograd_3x3_2x2_32x32",   "params": "updatw", "share": "(512*4 + 32)*4 + 8", "args": {"type": "s"}},
    "hconv_winograd_3x3_2x2_32x32":   {"threads": 256, "sass": "xconv_winograd_3x3_2x2_32x32",   "params": "updatw", "share": "(512*4 + 32)*4 + 8", "args": {"type": "h"}},

    "sconv_winograd_4x4_3x3_32x32":   {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32",   "params": "fpropw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_4x4_3x3_32x32":   {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32",   "params": "fpropw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},
    "sconv_winograd_4x4_3x3_32x32_X": {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32_X", "params": "fpropw4X", "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_4x4_3x3_32x32_X": {"threads": 640, "sass": "xconv_winograd_4x4_3x3_32x32_X", "params": "fpropw4X", "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},
    "sconv_winograd_3x3_4x4_32x32":   {"threads": 640, "sass": "xconv_winograd_3x3_4x4_32x32",   "params": "updatw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_3x3_4x4_32x32":   {"threads": 640, "sass": "xconv_winograd_3x3_4x4_32x32",   "params": "updatw4",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},

    "sconv_winograd_2x2_5x5_32x32":   {"threads": 640, "sass": "xconv_winograd_2x2_5x5_32x32",   "params": "fpropw5",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "s"}},
    "hconv_winograd_2x2_5x5_32x32":   {"threads": 640, "sass": "xconv_winograd_2x2_5x5_32x32",   "params": "fpropw5",  "share": "32*36*2*4 + 64 + 8", "args": {"type": "h"}},


    "sgemm_nn_128x128": {"threads": 256, "sass": "sgemm_nn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_nt_128x128": {"threads": 256, "sass": "sgemm_nt_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "sgemm_tn_128x128": {"threads": 256, "sass": "sgemm_tn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_nn_128x128": {"threads": 256, "sass": "hgemm_nn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_nt_128x128": {"threads": 256, "sass": "hgemm_nt_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},
    "hgemm_tn_128x128": {"threads": 256, "sass": "hgemm_tn_128x128", "params": "gemm", "share": "128*8*2 + 128*8*2 + 4"},

    "sgemm_nn_128x64":  {"threads": 128, "sass": "sgemm_nn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},
    "sgemm_tn_128x64":  {"threads": 128, "sass": "sgemm_tn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},
    "hgemm_nn_128x64":  {"threads": 128, "sass": "hgemm_nn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},
    "hgemm_tn_128x64":  {"threads": 128, "sass": "hgemm_tn_128x64",  "params": "gemm", "share": "128*8*2 +  64*8*2 + 4"},

    "sgemm_nn_128x32":  {"threads": 128, "sass": "sgemm_nn_128x32",  "params": "gemm", "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_tn_128x32":  {"threads": 128, "sass": "sgemm_tn_128x32",  "params": "gemm", "share": "(128*16 +  0)*2 + 32*16*2 + 4"},
    "hgemm_nn_128x32":  {"threads": 128, "sass": "hgemm_nn_128x32",  "params": "gemm", "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "hgemm_tn_128x32":  {"threads": 128, "sass": "hgemm_tn_128x32",  "params": "gemm", "share": "(128*16 +  0)*2 + 32*16*2 + 4"},

    "sgemm_nn_32x128":  {"threads": 128, "sass": "sgemm_nn_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4"},
    "sgemm_nt_32x128":  {"threads": 128, "sass": "sgemm_nt_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4"},
    "hgemm_nn_32x128":  {"threads": 128, "sass": "hgemm_nn_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 +  0)*2 + 4"},
    "hgemm_nt_32x128":  {"threads": 128, "sass": "hgemm_nt_32x128",  "params": "gemm", "share": "(32*16 + 32)*2 + (128*16 + 32)*2 + 4"},

    "hgemm_nt_32x32": {"threads": 128, "sass": "hgemm_nt_32x32", "params": "gemm", "share": "32*65*4 + 4" },
    "hgemm_nt_16x64": {"threads": 128, "sass": "hgemm_nt_16x64", "params": "gemm", "share": "(16*64 + 32)*2 + (64*64 + 32)*2 + 4" },
    "hgemm_nn_32x64": {"threads": 128, "sass": "hgemm_nn_32x64", "params": "gemm", "share": "32*33*2 + 64*32*2 + 2048" },  #artificially limit occpancy
    "hgemm_nn_16x64": {"threads": 128, "sass": "hgemm_nn_16x64", "params": "gemm", "share": "(16*64 + 32)*2 + 64*64*2 + 4" },

    "sgemm_rnn_nn_128x32":    {"threads": 128, "sass": "sgemm_nn_rnn_128x32",       "params": "gemm_rnn",   "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_rnn_nn_vec_128x32":    {"threads": 128, "sass": "sgemm_nn_rnn_128x32",       "params": "gemm_rnn",   "share": "(128*16 + 32)*2 + 32*16*2 + 4", "args": {"vec": "1"}},

    "sgemm_rnn_bprop_tn_128x32":    {"threads": 128, "sass": "sgemm_tn_rnn_bprop_128x32",       "params": "gemm_rnn_bprop",   "share": "(128*16 + 32)*2 + 32*16*2 + 4"},
    "sgemm_rnn_bprop_tn_vec_128x32":    {"threads": 128, "sass": "sgemm_tn_rnn_bprop_128x32",       "params": "gemm_rnn_bprop",   "share": "(128*16 + 32)*2 + 32*16*2 + 4", "args": {"vec": "1"}},

    "persistent_rnn_fprop": {"threads": 256, "sass": "persistent_rnn_fprop", "params": "rnn_fprop", "share": "(64*48) + 4"},
    "persistent_rnn_bprop": {"threads": 256, "sass": "persistent_rnn_bprop", "params": "rnn_bprop", "share": "(64*48) + 4"},
}
# 

N, C, K, \
D, H, W, \
T, R, S, \
pad_d, pad_h, pad_w, \
str_d, str_h, str_w, \
dil_d, dil_h, dil_w = [int(v) for v in sys.argv[1:]]

#query which kernel is used in this case
be = gen_backend('gpu', batch_size=N)
conv_layer = Convolution((1, R, S, K),
    strides={'str_d':str_d, 'str_h':str_h, 'str_w':str_w},
    padding={'pad_d':pad_d, 'pad_h':pad_h, 'pad_w':pad_w},
    dilation={'dil_d':dil_d, 'dil_h':dil_h, 'dil_w':dil_w})
conv_layer.configure((C, 1, H, W))
fprop_kernels = conv_layer.nglayer.fprop_kernels

# kernel name
kernel_name = fprop_kernels.kernel_name

# kernel args
# [grid block] 
kernel_args = fprop_kernels.kernel_args


