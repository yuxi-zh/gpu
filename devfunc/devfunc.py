# ******************************************************************************
# Copyright 2014-2018 Intel Corporation
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
import re
import sys
import os.path
import argparse
import subprocess
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.tools import context_dependent_memoize
from neon.initializers import Gaussian
from neon.layers import Convolution
from neon.backends import gen_backend

cache_dir = os.environ.get("DEVFUNC_CACHE_DIR")
base_dir  = os.path.dirname(__file__)
maxas_dir = os.path.join(base_dir, "kernels", "maxas")
sass_dir  = os.path.join(base_dir, "kernels", "sass")
remap_sass_dir = os.path.join(cache_dir, "kernels", "sass")
if not os.path.exists(remap_sass_dir):
    os.makedirs(remap_sass_dir)
cuda_dir  = os.path.join(cache_dir, "kernels", "cuda")
if not os.path.exists(cuda_dir):
    os.makedirs(cuda_dir)
cubin_dir = os.path.join(cache_dir, "kernels", "cubin")
if not os.path.exists(cubin_dir):
    os.makedirs(cubin_dir)

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

_params = {
    "fprop": [
        "float* param_Sum",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_N",
        "unsigned param_K",
        "unsigned param_D",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_WN",
        "unsigned param_HWN",
        "unsigned param_DHWN",
        "unsigned param_C",
        "unsigned param_KRST",
        "unsigned param_RST",
        "unsigned param_RS",
        "unsigned param_T",
        "unsigned param_R",
        "unsigned param_S",
        "unsigned param_magic_RS",
        "unsigned param_shift_RS",
        "unsigned param_magic_S",
        "unsigned param_shift_S",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_str_d",
        "unsigned param_str_h",
        "unsigned param_str_w",
        "unsigned param_dil_d",
        "unsigned param_dil_h",
        "unsigned param_dil_w",
        "unsigned param_P2",
        "unsigned param_Q",
        "unsigned param_PQk",
        "unsigned param_Qk",
        "unsigned param_k",
        "unsigned param_magic_PQk",
        "unsigned param_shift_PQk",
        "unsigned param_magic_Qk",
        "unsigned param_shift_Qk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_MPQN",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
        "unsigned param_gridMPQN",
    ],
    "fprop2": [
        "float* param_Sum",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_D",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_N",
        "unsigned param_K",
        "unsigned param_M",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_str_d",
        "unsigned param_str_h",
        "unsigned param_str_w",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_dil_d",
        "unsigned param_dil_h",
        "unsigned param_dil_w",
        "unsigned param_DHWN",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_MPQN",
        "unsigned param_PQN",
        "unsigned param_QN",
        "unsigned param_PQnk",
        "unsigned param_Qnk",
        "unsigned param_nk",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_magic_PQnk",
        "unsigned param_shift_PQnk",
        "unsigned param_magic_Qnk",
        "unsigned param_shift_Qnk",
        "unsigned param_magic_nk",
        "unsigned param_shift_nk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_Km32",
        "unsigned param_K32p",
        "unsigned param_TRSK",
        "unsigned param_TRS",
        "unsigned param_RS",
        "unsigned param_S",
        "unsigned param_magic_RS",
        "unsigned param_shift_RS",
        "unsigned param_magic_S",
        "unsigned param_shift_S",
        "unsigned param_gridP2",
        "unsigned param_gridQ",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
        "unsigned param_gridMPQN",
        "unsigned param_superM",
        "unsigned param_superP",
        "unsigned param_superQ",
        "unsigned param_superN",
        "unsigned param_shiftM",
        "unsigned param_shiftP",
        "unsigned param_shiftQ",
        "unsigned param_shiftN",
        "unsigned param_SuperM",
        "unsigned param_SuperP",
        "unsigned param_SuperQ",
        "unsigned param_SuperN",
    ],
    "updat2": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "unsigned param_C",
        "unsigned param_D",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_N",
        "unsigned param_K",
        "unsigned param_M",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_str_d",
        "unsigned param_str_h",
        "unsigned param_str_w",
        "int param_pad_d",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_dil_d",
        "unsigned param_dil_h",
        "unsigned param_dil_w",
        "unsigned param_DHWN",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_MPQN16p",
        "unsigned param_MPQN",
        "unsigned param_PQN",
        "unsigned param_QN",
        "unsigned param_PQkc",
        "unsigned param_Qkc",
        "unsigned param_kc",
        "unsigned param_c",
        "unsigned param_k",
        "unsigned param_magic_PQkc",
        "unsigned param_shift_PQkc",
        "unsigned param_magic_Qkc",
        "unsigned param_shift_Qkc",
        "unsigned param_magic_kc",
        "unsigned param_shift_kc",
        "unsigned param_magic_c",
        "unsigned param_shift_c",
        "unsigned param_CRSTK",
        "unsigned param_CRST",
        "unsigned param_TRS",
        "unsigned param_RS",
        "unsigned param_S",
        "unsigned param_magic_TRS",
        "unsigned param_shift_TRS",
        "unsigned param_magic_RS",
        "unsigned param_shift_RS",
        "unsigned param_magic_S",
        "unsigned param_shift_S",
        "unsigned param_superM",
        "unsigned param_superP",
        "unsigned param_superQ",
        "unsigned param_superN",
        "unsigned param_shiftM",
        "unsigned param_shiftP",
        "unsigned param_shiftQ",
        "unsigned param_strideP",
        "unsigned param_strideQ",
        "unsigned param_stridePQ",
        "unsigned param_gridP",
        "unsigned param_gridQ",
        "unsigned param_loopX",
        "unsigned param_loopXp",
        "unsigned param_loopQ",
        "unsigned param_loopQp",
        "unsigned param_loopN",
        "unsigned param_loopNp",
    ],
    "gemm": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_lda",
        "unsigned param_ldb",
        "unsigned param_ldc",
        "unsigned param_m",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_ldaz",
        "unsigned param_ldbz",
        "unsigned param_ldcz",
        "unsigned param_batch_loops",
    ],
    "gemm_rnn": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float* param_bias",
        "float* param_lock",
        "float param_alpha",
        "float param_beta",
        "float param_xcutoff",
        "int   param_flags",
        "int   param_lda",
        "int   param_ldb",
        "int   param_ldc",
        "int   param_m",
        "int   param_n",
        "int   param_k",
        "int   param_ldaz",
        "int   param_ldbz",
        "int   param_ldcz",
        "int   param_batch_loops",
        "int   param_dimB",
        "int   param_dimC",
        "int   param_unrolling",
        "int   param_numBlks",
        "int   param_numAblks"
    ],
    "gemm_rnn_bprop": [
        "float* param_C",
        "float* param_A",
        "float* param_B",
        "float* param_H",
        "float* param_lock",
        "float param_alpha",
        "float param_beta",
        "float param_xcutoff",
        "int   param_flags",
        "int   param_lda",
        "int   param_ldb",
        "int   param_ldc",
        "int   param_ldh",
        "int   param_m",
        "int   param_n",
        "int   param_k",
        "int   param_ldaz",
        "int   param_ldbz",
        "int   param_ldcz",
        "int   param_batch_loops",
        "int   param_dimB",
        "int   param_dimC",
        "int   param_dimH",
        "int   param_unrolling",
        "int   param_numBlks",
        "int   param_numAblks"
    ],
    "fpropw": [
        "float* param_S",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_H",
        "unsigned param_P",
        "int param_pad_h",
        "int param_pad_w",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_PQN",
        "unsigned param_QN",
        "unsigned param_Qnk",
        "unsigned param_nk",
        "unsigned param_n",
        "unsigned param_k",
        "unsigned param_magic_Qnk",
        "unsigned param_shift_Qnk",
        "unsigned param_magic_nk",
        "unsigned param_shift_nk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_RSK",
        "unsigned param_4RSKp",
        "unsigned param_4HWNp",
        "unsigned param_gridK",
        "unsigned param_gridP2",
        "unsigned param_gridQ",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
        "unsigned param_superP",
        "unsigned param_superQ",
        "unsigned param_superN",
        "unsigned param_shiftP",
        "unsigned param_shiftQ",
        "unsigned param_shiftN",
    ],
    "fpropw4X": [
        "float* param_S",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "unsigned param_Xk",
        "unsigned param_k",
        "unsigned param_magic_Xk",
        "unsigned param_shift_Xk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_C_1152",
        "unsigned param_GXS_C_1152",
        "unsigned param_GYS_GXS_C_1152",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_PQN15",
        "unsigned param_maskN",
        "unsigned param_shiftX",
        "unsigned param_shiftY",
        "unsigned param_superX",
        "unsigned param_superY",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
    ],
    "fpropw4": [
        "float* param_S",
        "float* param_X",
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "float param_beta",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "unsigned param_Y",
        "unsigned param_W",
        "unsigned param_YXN",
        "unsigned param_XN",
        "unsigned param_Y2",
        "unsigned param_GX",
        "unsigned param_Xk",
        "unsigned param_k",
        "unsigned param_magic_Xk",
        "unsigned param_shift_Xk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_PQN15",
        "unsigned param_maskN",
        "unsigned param_shiftX",
        "unsigned param_shiftY",
        "unsigned param_superX",
        "unsigned param_superY",
        "int param_pad_x",
        "int param_pad_y",
        "unsigned param_RSK",
        "unsigned param_RSK2p",
        "unsigned param_YXN2p",
        "unsigned param_gridN",
        "unsigned param_gridQN",
        "unsigned param_gridPQN",
    ],
    "fpropw5": [
        "float* param_O",
        "float* param_I",
        "float* param_F",
        "float param_alpha",
        "unsigned param_flags",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "unsigned param_H",
        "unsigned param_W",
        "unsigned param_HWN",
        "unsigned param_WN",
        "unsigned param_Y2",
        "unsigned param_GX",
        "unsigned param_Xk",
        "unsigned param_k",
        "unsigned param_magic_Xk",
        "unsigned param_shift_Xk",
        "unsigned param_magic_k",
        "unsigned param_shift_k",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_PQNp",
        "unsigned param_PQN15p",
        "unsigned param_shiftY",
        "unsigned param_shiftX",
        "unsigned param_shiftN",
        "unsigned param_superY",
        "unsigned param_superX",
        "unsigned param_superN",
        "unsigned param_SuperY",
        "unsigned param_SuperX",
        "unsigned param_SuperN",
        "int param_pad_x",
        "int param_pad_y",
        "unsigned param_HWN2p",
        "unsigned param_C_1152",
    ],
    "updatw": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "unsigned param_Y",
        "unsigned param_X",
        "unsigned param_P",
        "unsigned param_Q",
        "unsigned param_C",
        "unsigned param_K",
        "unsigned param_N",
        "int param_pad_y",
        "int param_pad_x",
        "unsigned param_GY",
        "unsigned param_GX",
        "unsigned param_GYS",
        "unsigned param_GXS",
        "unsigned param_shiftYI",
        "unsigned param_shiftXI",
        "unsigned param_superYI",
        "unsigned param_superXI",
        "unsigned param_superNI",
        "unsigned param_shiftY",
        "unsigned param_shiftX",
        "unsigned param_superY",
        "unsigned param_superX",
        "unsigned param_superN",
        "unsigned param_loopXI",
        "unsigned param_loopX",
        "unsigned param_loopN",
        "unsigned param_strideY",
        "unsigned param_strideX",
        "unsigned param_XN",
        "unsigned param_YXN",
        "unsigned param_QN",
        "unsigned param_PQN",
        "unsigned param_SK",
        "unsigned param_RSK",
        "unsigned param_Np",
        "unsigned param_XNp",
        "unsigned param_2XNp",
        "unsigned param_QNp",
        "unsigned param_CPQkc",
        "unsigned param_PQkc",
        "unsigned param_Qkc",
        "unsigned param_kc",
        "unsigned param_c",
        "unsigned param_k",
        "unsigned param_magic_CPQkc",
        "unsigned param_shift_CPQkc",
        "unsigned param_magic_PQkc",
        "unsigned param_shift_PQkc",
        "unsigned param_magic_Qkc",
        "unsigned param_shift_Qkc",
        "unsigned param_magic_kc",
        "unsigned param_shift_kc",
        "unsigned param_magic_c",
        "unsigned param_shift_c",
        "unsigned param_CRSK",
    ],
    "updatw4": [
        "float* param_F",
        "float* param_I",
        "float* param_E",
        "float param_alpha",
        "unsigned param_K",
        "unsigned param_C",
        "unsigned param_k",
        "unsigned param_c",
        "unsigned param_kc",
        "unsigned param_magic_kc",
        "unsigned param_shift_kc",
        "unsigned param_magic_c",
        "unsigned param_shift_c",
        "unsigned param_YXN2",
        "unsigned param_sYXN",
        "unsigned param_magic_sYXN",
        "unsigned param_shift_sYXN",
        "unsigned param_stride_YXNp",
        "unsigned param_YXN",
        "unsigned param_YXN_1152",
        "unsigned param_RSK",
        "unsigned param_CRSK",
        "unsigned param_Kp",
        "unsigned param_SKp",
        "unsigned param_RSK15_SK2p",
    ],
    "rnn_fprop": [
        "float* param_h",
        "float* param_hprev",
        "float* param_bias",
        "float* param_w",
        "int* param_lockAddr",
        "int param_ldh",
        "int param_ldw",
        "int param_bsz",
        "int param_seqLength",
        "int param_numBlks",
        "int param_rowSize",
        "int param_reverse",
        "float param_reluclip"
    ]
    ,
    "rnn_bprop": [
        "float* param_d",
        "float* param_dnext",
        "float* param_h",
        "float* param_w",
        "int* param_lockAddr",
        "int param_ldd",
        "int param_ldh",
        "int param_ldw",
        "int param_bsz",
        "int param_seqLength",
        "int param_numBlks",
        "int param_rowSize",
        "int param_reverse",
        "float param_reluclip"
    ]
}

_params["bprop"] = _params["fprop"] + [
        "unsigned param_magic_str_d",
        "unsigned param_shift_str_d",
        "unsigned param_magic_str_h",
        "unsigned param_shift_str_h",
        "unsigned param_magic_str_w",
        "unsigned param_shift_str_w",
    ]
_params["bprop2"] = _params["fprop2"] + [
        "unsigned param_magic_str_d",
        "unsigned param_shift_str_d",
        "unsigned param_magic_str_h",
        "unsigned param_shift_str_h",
        "unsigned param_magic_str_w",
        "unsigned param_shift_str_w",
    ]

_share_template = """
__shared__ float shared_memory[{}]; // shared memory allocation
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < param_MPQN) {{
    shared_memory[i] = param_I[threadIdx.x];
    param_O[threadIdx.x] = shared_memory[i];
}}
"""

_devfunc_template = """
extern "C" {{
__device__ 
void {} // device function name
({}) // device function parameters 
{{
 {}
}}
}}
"""

def get_cuda_file(kernel_name):
    kernel_args = ",\n".join(_params[kernels[kernel_name]["params"]])

    if "share" in kernels[kernel_name]:
        print(eval(kernels[kernel_name]["share"]))
        share = _share_template.format(eval(kernels[kernel_name]["share"]))
    else:
        share = ""

    cuda_text = _devfunc_template.format(kernel_name, kernel_args, share)
    cuda_name = kernel_name + ".cu"
    cuda_file = os.path.join(cuda_dir, cuda_name)
    
    with open(cuda_file, "w") as cuda:
        cuda.write(cuda_text);
    return cuda_file

include_re = re.compile(r'^<INCLUDE\s+file="([^"]+)"\s*/>')

def extract_includes(name, includes=None):
    if not includes:
        includes = list()
    sass_file = os.path.join(sass_dir, name)
    includes.append((sass_file, os.path.getmtime(sass_file)))
    for line in open(sass_file, "r"):
        match = include_re.search(line)
        if match:
            extract_includes(match.group(1), includes)
    return includes

def run_command(cmdlist):
    cmd  = " ".join(cmdlist)

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError("Error(%d):\n%s\n%s" % (proc.returncode, cmd, err))

def remap_address(sass_file, params, addrs):

    with open(sass_file, 'r') as sass:
        sass_text = sass.read()
    for param, addr in zip(params, addrs):
        _type, name = param.split()
        if '*' not in _type:
            map_template = r'{0}\s+:\s+c\[0x.*\]\[0x.*\]'
            new_mapping = '{}\t:\tc[0x0][{}]'.format(name, hex(addr))
            sass_text = re.sub(map_template.format(name), new_mapping, sass_text)
        else:
            map_template = r'{0}\[{1}\]\s+:\s+c\[0x.*\]\[0x.*\]'
            new_mapping_tmpalte = '{}[{}]\t:\tc[0x0][{}]'
            new_map0 = new_mapping_tmpalte.format(name, 0, hex(addr))
            new_map1 = new_mapping_tmpalte.format(name, 1, hex(addr+4))
            sass_text = re.sub(map_template.format(name, 0), new_map0, sass_text)
            sass_text = re.sub(map_template.format(name, 1), new_map1, sass_text)
    new_sass_file = os.path.join(remap_sass_dir, os.path.basename(sass_file))
    with open(new_sass_file, 'w') as sass:
        sass.write(sass_text);

    return new_mapping

def cumodule(base_name, param_addr, vblock=None, options=None):

    params = _params[kernels[base_name]['params']]
    assert len(params) == len(param_addr)

    attributes = drv.Context.get_device().get_attributes()
    major = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MAJOR]
    minor = attributes[drv.device_attribute.COMPUTE_CAPABILITY_MINOR]
    if major < 5:
        raise RuntimeError("sass kernels require Maxwell or greater class hardware")

    arch = "-arch compute_%d%d" % (major, minor)
    code = "-code sm_%d%d" % (major, minor)

    libprefix = "PERL5LIB=%s" % (maxas_dir)
    maxas_i = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -i -w"]
    maxas_p = [libprefix, os.path.join(maxas_dir, "maxas.pl") + " -p"]

    kernel_spec = kernels[base_name]
    kernel_name = base_name

    if "args" in kernel_spec:
        for pair in kernel_spec["args"].items():
            maxas_i.append("-D%s %s" % pair)
            maxas_p.append("-D%s %s" % pair)

    if options is not None:
        for opt in options:
            if type(opt) is tuple:
                maxas_i.append("-D%s %s" % opt)
                maxas_p.append("-D%s %s" % opt)
                kernel_name += "_%s%s" % opt
            else:
                maxas_i.append("-D%s 1" % opt)
                maxas_p.append("-D%s 1" % opt)
                kernel_name += "_%s" % opt

    maxas_i.insert(2, "-k " + kernel_name)

    sass_name  = kernel_spec["sass"] + ".sass"
    cubin_name = kernel_name + ".cubin"

    sass_file  = os.path.join(sass_dir, sass_name)
    cubin_file = os.path.join(cubin_dir, cubin_name)

    if not os.path.exists(sass_file):
        raise RuntimeError("Missing: %s for kernel: %s" % (sass_name, kernel_name))

    if not os.path.exists(cubin_file):
        cuda_file = get_cuda_file(kernel_name)
        run_command(["nvcc -ccbin g++ -m64 -dc", arch, code, \
            "-cubin", cuda_file, "-o", cubin_file])
        assert os.path.exists(cubin_file)

    includes = extract_includes(sass_name)
    for include, include_age in includes:
        remap_address(include, params, param_addr)
        if vblock is not None:
            virtual_block(inc)
        run_command(maxas_i + [sass_file, cubin_file])

    return cubin_file


def signature(N, C, K, D, H, W, T, R, S, \
    pad_d, pad_h, pad_w, str_d, str_h, str_w, \
    dil_d, dil_h, dil_w):

    #query which kernel is used in this case
    be = gen_backend('gpu', batch_size=N)
    conv_layer = Convolution((1, R, S, K),
        strides={'str_d':str_d, 'str_h':str_h, 'str_w':str_w},
        padding={'pad_d':pad_d, 'pad_h':pad_h, 'pad_w':pad_w},
        dilation={'dil_d':dil_d, 'dil_h':dil_h, 'dil_w':dil_w})
    conv_layer.configure((C, 1, H, W))
    fprop_kernels = conv_layer.nglayer.fprop_kernels
    print(fprop_kernels.kernel_name)
    print(','.join(_params[kernels[fprop_kernels.kernel_name]['params']]))
    print(','.join([str(i) for i in fprop_kernels.kernel_args[0]]))
    print(','.join([str(i) for i in fprop_kernels.kernel_args[1]]))
    print(','.join([str(i) for i in fprop_kernels.kernel_args[2:]]))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('')
    subparsers = parser.add_subparsers(dest='subcommand')
    signature_parser = subparsers.add_parser('signature',
        help='query kernel siganture, output to stdout')
    signature_parser.add_argument('N', type=int, help='batch')
    signature_parser.add_argument('C', type=int, help='in channel')
    signature_parser.add_argument('K', type=int, help='out channel')
    signature_parser.add_argument('D', type=int, help='input depth')
    signature_parser.add_argument('H', type=int, help='input height')
    signature_parser.add_argument('W', type=int, help='input width')
    signature_parser.add_argument('T', type=int, help='filter depth')
    signature_parser.add_argument('R', type=int, help='filter height')
    signature_parser.add_argument('S', type=int, help='filter width')
    signature_parser.add_argument('pad_d', type=int, help='padding depth')
    signature_parser.add_argument('pad_h', type=int, help='padding height')
    signature_parser.add_argument('pad_w', type=int, help='padding width')
    signature_parser.add_argument('str_d', type=int, help='stride depth')
    signature_parser.add_argument('str_h', type=int, help='stride height')
    signature_parser.add_argument('str_w', type=int, help='stride width')
    signature_parser.add_argument('dil_d', type=int, help='dilation depth')
    signature_parser.add_argument('dil_h', type=int, help='dilation height')
    signature_parser.add_argument('dil_w', type=int, help='dilation width')
    cumodule_parser = subparsers.add_parser('cumodule',
        help='generate cubin module according to arguments, \
        output absolut path of cubin module to stdout')
    cumodule_parser.add_argument('kernel_name', type=str)
    cumodule_parser.add_argument('param_addr', type=int, nargs='+',
        help='paramters address in constant memory, \
        which must be larger than 0x140')
    cumodule_parser.add_argument('param_vblock', type=int,
        help='vblock address in constant memory, \
        which points to virtual block array in global memory')
    args = parser.parse_args()
    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('subcommand')](**kwargs)

