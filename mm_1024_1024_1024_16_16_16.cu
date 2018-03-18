extern "C" __global__ void mm__kernel0( float* __restrict__ C,  float* __restrict__ A,  float* __restrict__ B) {
  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];
  C[(((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y))] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 48)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16384)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16400)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16416)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16432)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32768)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32784)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32800)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32816)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49152)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49168)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49184)] = 0.000000e+00f;
  C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49200)] = 0.000000e+00f;
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    A_shared[((((int)threadIdx.x) * 16) + ((int)threadIdx.y))] = A[(((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * 1024) + ((int)threadIdx.y)) + (k_outer * 16))];
    A_shared[(((((int)threadIdx.x) * 16) + ((int)threadIdx.y)) + 256)] = A[((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * 1024) + ((int)threadIdx.y)) + (k_outer * 16)) + 16384)];
    A_shared[(((((int)threadIdx.x) * 16) + ((int)threadIdx.y)) + 512)] = A[((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * 1024) + ((int)threadIdx.y)) + (k_outer * 16)) + 32768)];
    A_shared[(((((int)threadIdx.x) * 16) + ((int)threadIdx.y)) + 768)] = A[((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * 1024) + ((int)threadIdx.y)) + (k_outer * 16)) + 49152)];
    B_shared[((((int)threadIdx.x) * 64) + ((int)threadIdx.y))] = B[((((((int)blockIdx.y) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (k_outer * 16384))];
    B_shared[(((((int)threadIdx.x) * 64) + ((int)threadIdx.y)) + 16)] = B[(((((((int)blockIdx.y) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (k_outer * 16384)) + 16)];
    B_shared[(((((int)threadIdx.x) * 64) + ((int)threadIdx.y)) + 32)] = B[(((((((int)blockIdx.y) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (k_outer * 16384)) + 32)];
    B_shared[(((((int)threadIdx.x) * 64) + ((int)threadIdx.y)) + 48)] = B[(((((((int)blockIdx.y) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (k_outer * 16384)) + 48)];
    __syncthreads();
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      C[(((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y))] = (C[(((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y))] + (A_shared[((((int)threadIdx.x) * 16) + k_inner)] * B_shared[(((int)threadIdx.y) + (k_inner * 64))]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16)] + (A_shared[((((int)threadIdx.x) * 16) + k_inner)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 16)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32)] + (A_shared[((((int)threadIdx.x) * 16) + k_inner)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 32)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 48)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 48)] + (A_shared[((((int)threadIdx.x) * 16) + k_inner)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 48)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16384)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16384)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 256)] * B_shared[(((int)threadIdx.y) + (k_inner * 64))]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16400)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16400)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 256)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 16)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16416)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16416)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 256)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 32)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16432)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 16432)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 256)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 48)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32768)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32768)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 512)] * B_shared[(((int)threadIdx.y) + (k_inner * 64))]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32784)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32784)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 512)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 16)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32800)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32800)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 512)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 32)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32816)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 32816)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 512)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 48)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49152)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49152)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 768)] * B_shared[(((int)threadIdx.y) + (k_inner * 64))]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49168)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49168)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 768)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 16)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49184)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49184)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 768)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 32)]));
      C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49200)] = (C[((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + 49200)] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + 768)] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + 48)]));
    }
  }
}

