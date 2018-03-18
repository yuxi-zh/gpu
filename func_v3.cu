
#include "utils.h"
#include "cuda.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

#define BLOCK_DIM_X		(16)
#define BLOCK_DIM_Y		(16)

extern __device__ void mm_kernel0(int n, const float* A, const float* B, const float *C); 

__device__ int syncCount = 0;

__device__ inline void __syncgrid()
{
	int numThreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
	syncCount = atomicAdd(&syncCount, 1);
	__threadfence();
	while (syncCount != numThreads);
	atomicCAS(&syncCount, numThreads, 0);
	__threadfence();
}

__device__ float* T = NULL;

__global__ void __func_v3__(int n, const float* A, const float* B, 
							const float* C, float* D)
{
	int rank = blockIdx.x == 0 && blockIdx.y == 0 
			&& threadIdx.x == 0 && threadIdx.y == 0;
	
	if (rank == 0) {
		T = (float *)malloc(sizeof(float) * n * n);
	} 
	__syncgrid();
	mm_kernel0(T, B, C);
	__syncgrid();
	mm_kernel0(D, A, T);

	if (rank == 0) {
		free(T);
	}
}

void func_v3(int n, const float* A, const float* B, const float* C, float* D)
{	
	ASSERT(n == 1024);
	ASSERT(n % BLOCK_DIM_X == 0 && n % BLOCK_DIM_Y == 0);

	int numBlocksPerSM = 0, blockSize = BLOCK_DIM_X * BLOCK_DIM_Y;
	CHECK_CUDA_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocksPerSM, __func_v3__, blockSize, 8 * blockSize));
	
	int numBlocks = numBlocksPerSM * CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
	ASSERT_MSG((n / BLOCK_DIM_X) *(n / BLOCK_DIM_Y) <= numBlocks,
		"exceed max active blocks on device");
	
	dim3 dimGrid(n / BLOCK_DIM_X, n / BLOCK_DIM_Y);
	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);

	__func_v3__<<<dimGrid, dimBlock>>>(n, A, B, C, D);
	CHECK_CUDA_CALL(cudaGetLastError());
}
