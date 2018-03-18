
#include "utils.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

extern __device__ mm(int n, const float* A, const float* B, const float *C);

__device__ float* T = NULL;

__global__ void __func_v3__(int n, const float* A, const float* B, 
							const float* C, float* D)
{
	int rank = blockIdx.x == 0 && blockIdx.y == 0 
			&& threadIdx.x == 0 && threadIdx.y == 0;
	
	if (rank == 0) {
		T = (float *)malloc(sizeof(float) * n * n);
	} 
	this_grid().sync();
	mm(n, B, C, T);
	this_grid().sync();
	mm(n, A, T, D);

	if (rank == 0) {
		free(T);
	}
}

void func_v4(int n, const float* A, const float* B, const float* C, float* D)
{
	ASSERT(n % BLOCK_DIM_X == 0 && n % BLOCK_DIM_Y == 0);

	int numBlocksPerSM = 0, blockSize = BLOCK_DIM_X * BLOCK_DIM_Y;
	CHECK_CUDA_CALL(cuOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks, __func_v3__, blockSize, 8 * blockSize))
	
	int numBlocks = numBlocksPerSM * CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
	ASSERT_MSG((n / BLOCK_DIM_X) * *(n / BLOCK_DIM_Y) <= numBlocks,
		"exceed max active blocks on device");
	
	dim3 dimGrid(n / BLOCK_DIM_X, n / BLOCK_DIM_Y);
	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_Y);

	__func_v3__<<<dimGrid, dimBlock>>>(n, A, B, C, D);
	CHECK_CUDA_CALL(cudaGetLastError());
}
