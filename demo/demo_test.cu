
#include "cuda_runtime.h"
#include "assert.h"
#include "utils.h"

extern __device__ void DeviceCpy(float *A, float *B, int n);

__global__ void GlobalCpy(float *A, float *B, int n) {
    DeviceCpy(A, B, n);
}


int main(int argc, char const *argv[]) {
	
	float *DA = NULL;
	float *DB = NULL;
	CHECK_CUDA_CALL(cudaMalloc((void **)&DA, sizeof(float) * 256));	
	CHECK_CUDA_CALL(cudaMalloc((void **)&DB, sizeof(float) * 256));

	float *HA = new float[256];
	float *HB = new float[256];
	
	for (int i = 0; i < 256; i++)
		HA[i] = i;
	
	CHECK_CUDA_CALL(cudaMemcpy(DA, HA, sizeof(float) * 256, cudaMemcpyHostToDevice));
	GlobalCpy<<<1, 256>>>(DA, DB, 256);
	CHECK_CUDA_CALL(cudaGetLastError());
	CHECK_CUDA_CALL(cudaMemcpy(HB, DB, sizeof(float) * 256, cudaMemcpyDeviceToHost));

	for (int i = 0; i < 256; i++)
		assert(HB[i] == i);

	CHECK_CUDA_CALL(cudaFree(DA));
	CHECK_CUDA_CALL(cudaFree(DB));
	
	delete[] HA;
	delete[] HB;

	return 0;
}
