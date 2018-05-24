extern "C" {
__device__ void sconv_direct_fprop_128x128_device(
    float* param_Sum,
    float* param_X,
    float* param_O,
    float* param_I,
    float* param_F,
    float param_alpha,
    float param_beta,
    unsigned param_flags,
    unsigned param_N,
    unsigned param_K,
    unsigned param_D,
    unsigned param_H,
    unsigned param_W,
    unsigned param_WN,
    unsigned param_HWN,
    unsigned param_DHWN,
    unsigned param_C,
    unsigned param_KRST,
    unsigned param_RST,
    unsigned param_RS,
    unsigned param_T,
    unsigned param_R,
    unsigned param_S,
    unsigned param_magic_RS,
    unsigned param_shift_RS,
    unsigned param_magic_S,
    unsigned param_shift_S,
    int param_pad_d,
    int param_pad_h,
    int param_pad_w,
    unsigned param_str_d,
    unsigned param_str_h,
    unsigned param_str_w,
    unsigned param_dil_d,
    unsigned param_dil_h,
    unsigned param_dil_w,
    unsigned param_P2,
    unsigned param_Q,
    unsigned param_PQk,
    unsigned param_Qk,
    unsigned param_k,
    unsigned param_magic_PQk,
    unsigned param_shift_PQk,
    unsigned param_magic_Qk,
    unsigned param_shift_Qk,
    unsigned param_magic_k,
    unsigned param_shift_k,
    unsigned param_QN,
    unsigned param_PQN,
    unsigned param_MPQN,
    unsigned param_gridN,
    unsigned param_gridQN,
    unsigned param_gridPQN,
    unsigned param_gridMPQN)
{
    __shared__ float shared_memory[4106];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < param_MPQN) {
        shared_memory[i] = param_I[threadIdx.x];
        param_O[threadIdx.x] = shared_memory[i];
    }
}

}
