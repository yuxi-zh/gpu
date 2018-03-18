produce D {
  // attr [iter_var(blockIdx.x, , blockIdx.x)] thread_extent = 16
  // attr [A.shared] storage_scope = "shared"
  allocate A.shared[float32 * 64 * 16]
  // attr [T] storage_scope = "local"
  allocate T[float32 * 1 * 49]
  // attr [B.shared] storage_scope = "shared"
  allocate B.shared[float32 * 16 * 16]
  // attr [C.shared] storage_scope = "shared"
  allocate C.shared[float32 * 1024]
  // attr [A.shared.local] storage_scope = "local"
  allocate A.shared.local[float32 * 4 * 1 * 1]
  // attr [T.shared.local] storage_scope = "local"
  allocate T.shared.local[float32 * 4 * 1 * 1]
  // attr [iter_var(blockIdx.y, , blockIdx.y)] thread_extent = 16
  // attr [iter_var(threadIdx.x, Range(min=0, extent=16), threadIdx.x)] thread_extent = 16
  // attr [iter_var(threadIdx.y, Range(min=0, extent=16), threadIdx.y)] thread_extent = 16
  D[(((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16384)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32768)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49152)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16400)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32784)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49168)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16416)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32800)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49184)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 48)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16432)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32816)] = 0.000000f
  D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49200)] = 0.000000f
  for (k.outer, 0, 64) {
    produce A.shared {
      for (ax0.outer, 0, 4) {
        A.shared[(((threadIdx.x*16) + threadIdx.y) + (ax0.outer*256))] = A[((((((blockIdx.x*64) + threadIdx.x)*1024) + threadIdx.y) + (k.outer*16)) + (ax0.outer*16384))]
      }
    }
    produce T {
      for (ty.init, 0, 49) {
        T[ty.init] = 0.000000f
      }
      for (l.outer, 0, 64) {
        produce B.shared {
          B.shared[((threadIdx.x*16) + threadIdx.y)] = B[((((threadIdx.x*1024) + threadIdx.y) + (k.outer*16384)) + (l.outer*16))]
        }
        produce C.shared {
          for (ax1.outer, 0, 4) {
            C.shared[(((threadIdx.x*64) + threadIdx.y) + (ax1.outer*16))] = C[(((((blockIdx.y + (threadIdx.x*16))*64) + threadIdx.y) + (l.outer*16384)) + (ax1.outer*16))]
          }
        }
        for (ty, 0, 49) {
          for (l.inner, 0, 16) {
            T[ty] = (T[ty] + (B.shared[((threadIdx.x*16) + l.inner)]*C.shared[((threadIdx.y + ty) + (l.inner*64))]))
          }
        }
      }
    }
    produce T.shared {
      for (ax1.outer, 0, 4) {
        C.shared[(((threadIdx.x*64) + threadIdx.y) + (ax1.outer*16))] = T[(ax1.outer*16)]
      }
    }
    for (k.inner, 0, 16) {
      produce A.shared.local {
        A.shared.local[0] = A.shared[((threadIdx.x*16) + k.inner)]
        A.shared.local[1] = A.shared[(((threadIdx.x*16) + k.inner) + 256)]
        A.shared.local[2] = A.shared[(((threadIdx.x*16) + k.inner) + 512)]
        A.shared.local[3] = A.shared[(((threadIdx.x*16) + k.inner) + 768)]
      }
      produce T.shared.local {
        T.shared.local[0] = C.shared[(threadIdx.y + (k.inner*64))]
        T.shared.local[1] = C.shared[((threadIdx.y + (k.inner*64)) + 16)]
        T.shared.local[2] = C.shared[((threadIdx.y + (k.inner*64)) + 32)]
        T.shared.local[3] = C.shared[((threadIdx.y + (k.inner*64)) + 48)]
      }
      D[(((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y)] = (D[(((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y)] + (A.shared.local[0]*T.shared.local[0]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16384)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16384)] + (A.shared.local[1]*T.shared.local[0]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32768)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32768)] + (A.shared.local[2]*T.shared.local[0]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49152)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49152)] + (A.shared.local[3]*T.shared.local[0]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16)] + (A.shared.local[0]*T.shared.local[1]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16400)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16400)] + (A.shared.local[1]*T.shared.local[1]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32784)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32784)] + (A.shared.local[2]*T.shared.local[1]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49168)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49168)] + (A.shared.local[3]*T.shared.local[1]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32)] + (A.shared.local[0]*T.shared.local[2]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16416)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16416)] + (A.shared.local[1]*T.shared.local[2]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32800)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32800)] + (A.shared.local[2]*T.shared.local[2]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49184)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49184)] + (A.shared.local[3]*T.shared.local[2]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 48)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 48)] + (A.shared.local[0]*T.shared.local[3]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16432)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 16432)] + (A.shared.local[1]*T.shared.local[3]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32816)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 32816)] + (A.shared.local[2]*T.shared.local[3]))
      D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49200)] = (D[((((((blockIdx.x*1024) + blockIdx.y) + (threadIdx.x*16))*64) + threadIdx.y) + 49200)] + (A.shared.local[3]*T.shared.local[3]))
    }
  }
}

extern "C" __global__ void K1__kernel0( float* __restrict__ D,  float* __restrict__ A,  float* __restrict__ B,  float* __restrict__ C) {
  __shared__ float A_shared[1024];
   float T[49];
  __shared__ float B_shared[256];
  __shared__ float C_shared[1024];
   float A_shared_local[4];
   float T_shared_local[4];
  D[(((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y))] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16384)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32768)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49152)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16400)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32784)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49168)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16416)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32800)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49184)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 48)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16432)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32816)] = 0.000000e+00f;
  D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49200)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      A_shared[(((((int)threadIdx.x) * 16) + ((int)threadIdx.y)) + (ax0_outer * 256))] = A[((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * 1024) + ((int)threadIdx.y)) + (k_outer * 16)) + (ax0_outer * 16384))];
    }
    for (int ty_init = 0; ty_init < 49; ++ty_init) {
      T[ty_init] = 0.000000e+00f;
    }
    for (int l_outer = 0; l_outer < 64; ++l_outer) {
      __syncthreads();
      B_shared[((((int)threadIdx.x) * 16) + ((int)threadIdx.y))] = B[((((((int)threadIdx.x) * 1024) + ((int)threadIdx.y)) + (k_outer * 16384)) + (l_outer * 16))];
      for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
        C_shared[(((((int)threadIdx.x) * 64) + ((int)threadIdx.y)) + (ax1_outer * 16))] = C[(((((((int)blockIdx.y) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (l_outer * 16384)) + (ax1_outer * 16))];
      }
      __syncthreads();
      for (int ty = 0; ty < 49; ++ty) {
        for (int l_inner = 0; l_inner < 16; ++l_inner) {
          T[ty] = (T[ty] + (B_shared[((((int)threadIdx.x) * 16) + l_inner)] * C_shared[((((int)threadIdx.y) + ty) + (l_inner * 64))]));
        }
      }
    }
    __syncthreads();
    for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
      C_shared[(((((int)threadIdx.x) * 64) + ((int)threadIdx.y)) + (ax1_outer1 * 16))] = T[(ax1_outer1 * 16)];
    }
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      A_shared_local[0] = A_shared[((((int)threadIdx.x) * 16) + k_inner)];
      A_shared_local[1] = A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 256)];
      A_shared_local[2] = A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 512)];
      A_shared_local[3] = A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 768)];
      T_shared_local[0] = C_shared[(((int)threadIdx.y) + (k_inner * 64))];
      T_shared_local[1] = C_shared[((((int)threadIdx.y) + (k_inner * 64)) + 16)];
      T_shared_local[2] = C_shared[((((int)threadIdx.y) + (k_inner * 64)) + 32)];
      T_shared_local[3] = C_shared[((((int)threadIdx.y) + (k_inner * 64)) + 48)];
      D[(((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y))] = (D[(((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y))] + (A_shared_local[0] * T_shared_local[0]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16384)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16384)] + (A_shared_local[1] * T_shared_local[0]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32768)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32768)] + (A_shared_local[2] * T_shared_local[0]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49152)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49152)] + (A_shared_local[3] * T_shared_local[0]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16)] + (A_shared_local[0] * T_shared_local[1]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16400)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16400)] + (A_shared_local[1] * T_shared_local[1]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32784)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32784)] + (A_shared_local[2] * T_shared_local[1]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49168)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49168)] + (A_shared_local[3] * T_shared_local[1]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32)] + (A_shared_local[0] * T_shared_local[2]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16416)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16416)] + (A_shared_local[1] * T_shared_local[2]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32800)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32800)] + (A_shared_local[2] * T_shared_local[2]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49184)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49184)] + (A_shared_local[3] * T_shared_local[2]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 48)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 48)] + (A_shared_local[0] * T_shared_local[3]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16432)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16432)] + (A_shared_local[1] * T_shared_local[3]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32816)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32816)] + (A_shared_local[2] * T_shared_local[3]));
      D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49200)] = (D[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49200)] + (A_shared_local[3] * T_shared_local[3]));
    }
  }
}


time: 642.737132 ms
