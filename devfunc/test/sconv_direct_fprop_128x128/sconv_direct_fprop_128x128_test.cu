
#include <assert.h>
#include <cuda.h>
#include <builtin_types.h>
#include "utils.h"

#include <tuple>
#include <bitset>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>

using namespace std;

#define __Stringlize__(literal) #literal
#define __Concatenate__(left, right) left ## right

#define Name(postfix) __Concatenate__(sconv_direct_fprop_128x128, postfix)
#define Stringlize(literal) __Stringlize__(literal)

// #define NeonFile Name(.cubin)

#define Signature \
    float* Sum, float* X, float* O, float* I, float* F, \
    float alpha, float beta, unsigned flags, \
    unsigned N, unsigned K, unsigned D, unsigned H, unsigned W, \
    unsigned WN, unsigned HWN, unsigned DHWN, unsigned C, \
    unsigned KRST, unsigned RST, unsigned RS, unsigned T, \
    unsigned R, unsigned S, \
    unsigned magic_RS, unsigned shift_RS, unsigned magic_S, unsigned shift_S, \
    int pad_d, int pad_h, int pad_w, \
    unsigned str_d, unsigned str_h, unsigned str_w, \
    unsigned dil_d, unsigned dil_h, unsigned dil_w, \
    unsigned P2, unsigned Q, unsigned PQk, unsigned Qk, unsigned k, \
    unsigned magic_PQk, unsigned shift_PQk, unsigned magic_Qk, unsigned shift_Qk, \
    unsigned magic_k, unsigned shift_k, \
    unsigned QN, unsigned PQN, unsigned MPQN, \
    unsigned gridNw, unsigned gridQNw, unsigned gridPQNw, unsigned gridMPQNw

#define GetVar(x) (x)
#define GetAddr(x) (&(x))
#define Dump(x) (cout << x << " ")

#define Parameter(O, Macro) \
    Macro(Sum), Macro(X), Macro(O), Macro(I), Macro(F), \
    Macro(alpha), Macro(beta), Macro(flags), \
    Macro(N), Macro(K), Macro(D), Macro(H), Macro(W), \
    Macro(WN), Macro(HWN), Macro(DHWN), \
    Macro(C), \
    Macro(KRST), Macro(RST), Macro(RS), Macro(T), Macro(R), Macro(S), \
    Macro(magic_RS), Macro(shift_RS), Macro(magic_S), Macro(shift_S), \
    Macro(pad_d), Macro(pad_h),Macro(pad_w), \
    Macro(str_d), Macro(str_h), Macro(str_w), \
    Macro(dil_d), Macro(dil_h), Macro(dil_w), \
    Macro(P2), Macro(Q), Macro(PQk), Macro(Qk), Macro(k), \
    Macro(magic_PQk), Macro(shift_PQk), Macro(magic_Qk), Macro(shift_Qk), \
    Macro(magic_k), Macro(shift_k), \
    Macro(QN), Macro(PQN), Macro(MPQN), \
    Macro(gridNw), Macro(gridQNw), Macro(gridPQNw), Macro(gridMPQNw)

extern "C" __device__ void Name(_device) (Signature);
__global__ void Name(_global) (Signature)
{
    Name(_device)(Parameter(O, GetVar));
}

void **MakeLaunchKernelParameter(initializer_list<void *> list)
{
    void **param = new void*[list.size()];
    void **param_iter = param;
    copy(list.begin(), list.end(), param_iter);
    return param;
}

unsigned ceil_div(unsigned x, unsigned y)
{
	return (x + y - 1) / y;
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
	unsigned nbits = (sizeof(unsigned) * 8) - 2;
	for (int p = 0; p < 2 * nbits + 1; p++) {
        unsigned long long t = 1ULL << p;
		if (t > nc * (d - 1 - (t - 1) % d)) {
			unsigned m = (t + d - 1 - (t - 1) % d) / d;
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

template <typename T>
class HDMem {
public:
    HDMem(size_t size) : size(size) {
        hbase = MakeRandomMemory();
        ASSERT(hbase != nullptr);
        CHECK_CUDA_CALL(cudaMalloc(&dbase, size * sizeof(T)));
        CHECK_CUDA_CALL(cudaMemcpy(
            dbase, hbase, size * sizeof(T), cudaMemcpyHostToDevice));
    }
    
    ~HDMem() {
        delete[] hbase;
        CHECK_CUDA_CALL(cudaFree(dbase));
    }

    HDMem(const HDMem&) = delete;
    HDMem& operator=(const HDMem&) = delete;
    
    operator T*() {
        return dbase;
    }

    void CopyBackToHost() {
        CHECK_CUDA_CALL(cudaMemcpy(
            hbase, dbase, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    bool operator==(const HDMem<T>& other) {
        bool close = true;
        for (size_t i = 0; i < size; i++) {
            if (abs(hbase[i] - other.hbase[i]) > 1e5) {
                close = false;
                break;
            }
        }
        return close;
    }

private:
    T *MakeRandomMemory() {
        T *array = new T[size];
        ifstream ifs("/dev/urandom", ifstream::binary);
        ASSERT(ifs.is_open());
        char *base = reinterpret_cast<char *>(array);
        size_t i, length = size * sizeof(T);
        for (i = 0; i < length; i+=1024) {
            ifs.read(base + i, 1024);
            ASSERT(ifs);
        }
        ifs.read(base + i - 1024, length - i + 1024);
        ASSERT(ifs);
        ifs.close();
        return array;
    }

private:
    size_t size;
    T *hbase;
    T *dbase;
};

int main(int argc, char const *argv[])
{
    unsigned N = 128, C = 4  , K = 128;
    unsigned D = 1  , H = 32, W = 32;
    unsigned T = 1  , R = 3  , S = 3;
    unsigned pad_d = 0, pad_h = 0, pad_w = 0;
    unsigned str_d = 1, str_h = 1, str_w = 1;
    unsigned dil_d = 1, dil_h = 1, dil_w = 1;
    unsigned M = 1;
    unsigned P = (H + 2 * pad_h - (dil_h * (R - 1) + 1)) / str_h + 1;
    unsigned Q = (W + 2 * pad_w - (dil_w * (S - 1) + 1)) / str_w + 1;
    unsigned WN = W * N, HWN = H * W * N, DHWN = D * H * W * N;
    unsigned QN = Q * N, PQN = P * Q * N, MPQN = M * P * Q * N;
    unsigned blockK = 128, blockN = 128;
    unsigned gridK = ceil_div(K, blockK), gridN = ceil_div(N, blockN);
    unsigned RS = R * S, RST = T * RS, KRST = K * RST;
    unsigned k = closest_divisor(gridK, 128 / blockK);
    unsigned P2 = P / 2, Q2 = Q * 2;
    unsigned Qk = Q2 * k, PQk = P * Q * k;
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
    float alpha = 1.0, beta = 0.0;
    unsigned flags = 0;
    float *Sum = nullptr, *X = nullptr;

    HDMem<float> I(N * D * H * W * C);
    HDMem<float> F(K * T * R * S * C);
    HDMem<float> CTRL(N * M * P * Q * K);
    HDMem<float> EXPR(N * M * P * Q * K);

    CHECK_CUDA_CALL(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 4096)); 

    Parameter(CTRL, Dump) << endl;;
    dim3 grid(gridMPQ * k, gridK / k, gridN), block(256, 1, 1);
    cout << "grid(" << grid.x <<"," << grid.y << "," << grid.z << ")" << endl;
    cout << "block(" << block.x <<"," << block.y << "," << block.z << ")" << endl;

    cout << "EXPR begin" << endl;
    Name(_global)<<<grid, block>>>(Parameter(EXPR, GetVar));
    CHECK_CUDA_CALL(cudaGetLastError());
    cout << "EXPR finished" << endl;

    cout << "CTRL begin" << endl;
    CUmodule module;
    CUfunction function;
    CUstream stream; 
    int shared_size;
    CHECK_CUDADriver_CALL(cuModuleLoad(
        &module, Stringlize(Name().cubin)));
    CHECK_CUDADriver_CALL(cuModuleGetFunction(
        &function, module, Stringlize(Name())));
    CHECK_CUDADriver_CALL(cuFuncGetAttribute(
        &shared_size, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
    CHECK_CUDADriver_CALL(cuStreamCreate(
        &stream, CU_STREAM_DEFAULT));
    CHECK_CUDADriver_CALL(cuLaunchKernel(
        function, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_size,
        stream, MakeLaunchKernelParameter({Parameter(CTRL, GetAddr)}), nullptr));
    cout << "CTRL finished" << endl;

    CTRL.CopyBackToHost();
    EXPR.CopyBackToHost();
    ASSERT(CTRL == EXPR);

	return 0;
}