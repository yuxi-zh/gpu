
#include <cuda_runtime.h>
#include "utils.h"

#include <tuple>
#include <bitset>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace std;

extern __device__ void sconv_direct_fprop_128x128(        
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
    unsigned param_gridMPQN);

__global__ void sconv_direct_fprop_128x128_global(
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
	sconv_direct_fprop_128x128(
		param_Sum,
    	param_X,
    	param_O,
    	param_I,
    	param_F,
    	param_alpha,
    	param_beta,
    	param_flags,
    	param_N,
    	param_K,
    	param_D,
    	param_H,
    	param_W,
    	param_WN,
    	param_HWN,
    	param_DHWN,
    	param_C,
    	param_KRST,
    	param_RST,
    	param_RS,
    	param_T,
    	param_R,
    	param_S,
    	param_magic_RS,
    	param_shift_RS,
    	param_magic_S,
    	param_shift_S,
    	param_pad_d,
    	param_pad_h,
    	param_pad_w,
    	param_str_d,
    	param_str_h,
    	param_str_w,
    	param_dil_d,
    	param_dil_h,
    	param_dil_w,
    	param_P2,
    	param_Q,
    	param_PQk,
    	param_Qk,
    	param_k,
    	param_magic_PQk,
    	param_shift_PQk,
    	param_magic_Qk,
    	param_shift_Qk,
    	param_magic_k,
    	param_shift_k,
    	param_QN,
    	param_PQN,
    	param_MPQN,
    	param_gridN,
    	param_gridQN,
    	param_gridPQN,
    	param_gridMPQN);
}

unsigned ceil_div(unsigned x, unsigned y)
{
	return -(-x / y);
}

unsigned closest_divisor(unsigned val, unsigned div)
{
	vector<pair<signed, signed> > list;
	for (int i = 1; i < 8; i++) {
		if (val % i == 0) {
			list.push_back(pair<signed, signed>(i - div, -div));
		}
	}
	sort(list.begin(), list.end());
	return -(list[0].second);
}

tuple<unsigned, unsigned> magic32(unsigned nmax, unsigned d)
{
	unsigned nc = ((nmax + 1) / d) * d - 1;
	unsigned nbits = bitset<32>(nmax).to_string().size() - 2;
	for (int p = 0; p < 2 * nbits - 2; p++) {
		if ((1 << p) > nc * (d - 1 - (1 << (p - 1)) % d)) {
			unsigned m = ((1 << p) + d - 1 - (1 << (p - 1)) % d);
			return tuple<unsigned, unsigned>(m, p);
		}
	}
	throw runtime_error("Can't find magic number for division");
}

tuple<unsigned, unsigned> magic64(unsigned d)
{
	unsigned nmax, magic, shift;

	if (d == 3) 
		nmax = 0xffffffff;
	else
		nmax = 0x7fffffff;

	tie(magic, shift) = magic32(nmax, d);
	if (magic != 1)
		shift -= 32;
	return tuple<unsigned, unsigned>(magic, shift);
}

int main(int argc, char const *argv[])
{
    unsigned N = 128;
    unsigned C = 4;
    unsigned K = 64;
    unsigned D = 1;
    unsigned H = 128;
    unsigned W = 128;
    unsigned T = 1;
    unsigned R = 3;
    unsigned S = 3;
    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;
    unsigned str_d = 1;
    unsigned str_h = 1;
    unsigned str_w = 1;
    unsigned dil_d = 1;
    unsigned dil_h = 1;
    unsigned dil_w = 1;
    unsigned M = 1;
    unsigned P = (H + 2 * pad_h - (dil_h * (R - 1) + 1)) / str_h + 1;
    unsigned Q = (W + 2 * pad_w - (dil_w * (S - 1) + 1)) / str_w + 1;

    float* Sum = NULL;
    float* X = NULL;
    float* O = new float[N * M * P * Q * K];
    float* I = new float[N * D * H * W * C];
    float* F = new float[K * T * R * S * C];
    float alpha = 1.0;
    float beta = 0.0;
    unsigned flags = 0;

    float* DO = NULL;
    float* DI = NULL;
    float* DF = NULL;

    CHECK_CUDA_CALL(cudaMalloc((void **)&DO, sizeof(float) * N * M * P * Q * K));
    CHECK_CUDA_CALL(cudaMalloc((void **)&DI, sizeof(float) * N * D * H * W * C));
    CHECK_CUDA_CALL(cudaMalloc((void **)&DF, sizeof(float) * K * T * R * S * C));

    CHECK_CUDA_CALL(cudaMemcpy(DI, I, sizeof(float) * N * D * H * W * C, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(DF, F, sizeof(float) * K * T * R * S * C, cudaMemcpyHostToDevice));

    unsigned blockK = 128;
    unsigned blockN = 128;

    unsigned gridK = ceil_div(K, blockK);
    unsigned gridN = ceil_div(N, blockN);
    
    unsigned RS = R * S;
    unsigned RST = T * RS;
    unsigned KRST = K * RST;

    unsigned k = closest_divisor(gridK, 128 / blockK);

    unsigned P2 = P / 2;
    unsigned Q2 = Q * 2;
    unsigned Qk = Q2 * k;
    unsigned PQk = P * Q * k;
    
    unsigned magic_PQk, shift_PQk;
    tie(magic_PQk, shift_PQk) = magic64(PQk);
    unsigned magic_Qk, shift_Qk;
    tie(magic_Qk, shift_Qk) = magic64(Qk);
    unsigned magic_k, shift_k;
    tie(magic_k, shift_k) = magic32(Qk, k);
    unsigned magic_RS, shift_RS;
    tie(magic_RS, shift_RS) = magic32(RST + 32, RS);
    unsigned magic_S, shift_S;
    tie(magic_S, shift_S) = magic32(RS + 32, S);

	unsigned bsum_warps = blockN / 64;
    unsigned gridNw = gridN * bsum_warps;
    unsigned gridQNw = Q * gridNw;
    unsigned gridPQNw = P * gridQNw;
	unsigned gridMPQNw = M * gridPQNw;
	unsigned gridMPQ = M * P * Q;    

	dim3 grid(gridMPQ * k, gridK / k, gridN);
	dim3 block(256, 1, 1);

    sconv_direct_fprop_128x128_global<<<grid, block>>>(
    	Sum,
    	X,
    	DO,
    	DI,
    	DF,
    	alpha,
    	beta,
    	flags,
    	N,
    	K,
    	D,
    	H,
    	W,
    	W * N,
    	H * W * N,
    	D * H * W * N,
    	C,
    	KRST,
    	RST,
    	RS,
    	T,
    	R,
    	S,
    	magic_RS,
    	shift_RS,
    	magic_S,
    	shift_S,
    	pad_d,
    	pad_h,
    	pad_w,
    	str_d,
    	str_h,
    	str_w,
    	dil_d,
    	dil_h,
    	dil_w,
    	P2,
    	Q,
    	PQk,
    	Qk,
    	k,
    	magic_PQk,
    	shift_PQk,
    	magic_Qk,
    	shift_Qk,
    	magic_k,
    	shift_k,
    	Q * N,
    	P * Q * N,
    	M * P * Q * N,
    	gridNw,
    	gridQNw,
    	gridPQNw,
    	gridMPQNw);
    CHECK_CUDA_CALL(cudaGetLastError());
    CHECK_CUDA_CALL(cudaMemcpy(O, DO, sizeof(float) * N * M * P * Q * K, cudaMemcpyDeviceToHost));

    CHECK_CUDA_CALL(cudaFree(DI));
    CHECK_CUDA_CALL(cudaFree(DO));
    CHECK_CUDA_CALL(cudaFree(DF));

    delete[] I;
    delete[] F;
    delete[] O;

	return 0;
}