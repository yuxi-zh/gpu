#include "stdio.h"
#include "stdlib.h"
#include "errno.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "helper_cuda.h"
#include "cooperative_groups.h"

using namespace cooperative_groups;

#define allocate_host_memory(name, size, errhanlde) 							\
	float *name = (float *)malloc(sizeof(float) * size); 						\
	if (name == nullptr) 														\
	{																			\
		fprintf(stderr, "Host memory allocation for %s failed: %s\n",			\
			#name, strerrno(errno));											\
		goto errhanlde;															\
	}

#define allocate_device_memory(name, size, errhanlde)							\
	if (cudaMalloc((void **)&name, size * sizeof(name[0]) != cudaSuccess)		\
	{																			\
		fprintf(stderr, "Device memory allocation for %s failed\n", #name);		\
		goto errhanlde;															\
	}

#define initialize_device_matrices(host, device, size, errhanlde)				\
	status = cublasSetVector(size, sizeof(host[0]), host, 1, device, 1)			\
	if (status != CUBLAS_STATUS_SUCCESS)										\
	{																			\
		fprintf(stderr, "device access from %s to %s error\n", #host, #device);	\
		goto errhanlde;															\
	}

#define free_device_memory(name, errhanlde)										\
	if ((error = cudaFree(name)) != cudaSuccess) {								\
		fprintf(stderr, "Memory free %s failed\n", #name);						\
		goto errhanlde;															\
	}

#define random_fill(M, n)														\
	for (int i = 0; i < n; ++i)													\
		M[i] = rand() / (float)RAND_MAX;										\

void threeMatrixMulV0(int n, const float *A, const float *B,
						const float *C, const float *D)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			D[i * n + j] = 0;
			for (int k = 0; k < n; ++k)
			{
				for (int l = 0; l < n; ++l)
				{
					D[i * n + j] += A[i *n + k] * B[k * n + l] * C[l * n + j];
				}
			}
		}
	}
}

void threeMatrixMulV1(int n, const float *A, const float *B, 
						const float *C, const float *D)
{
	cublasHanlde_t hanlde;
	cublasStatus_t status = cublasCreate(&hanlde);

	float *DT;
	allocate_device_memory(DT, SIZE, finish);

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 1.0f, B, n, C, n, 0.0f, T, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 1.0f, A, n, T, n, 0.0f, D, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;

finish:
	cublasDestory(hanlde);
}

__global__ void threeMatrixMulV2(cublasStatus_t *returnValue, int n,
									const float *A, const float *B,
									const float *C, const float *D,
									const float *T)
{
	cublasHanlde_t hanlde;
	cublasStatus_t status = cublasCreate(&hanlde);

	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 1.0f, B, n, C, n, 0.0f, T, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 1.0f, A, n, T, n, 0.0f, D, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;


finish:
	cublasDestory(hanlde);
	*returnValue = status;
}

__device__ void twoMatrixMul(float* __restrict__ A, float* __restrict__ B,
								float* __restrict__ C)
{

  __shared__ float A_shared[1024];
  __shared__ float B_shared[1024];

  for (int dx_inner_outer_init = 0; dx_inner_outer_init < 4; ++dx_inner_outer_init) {
    for (int dy_inner_outer_init = 0; dy_inner_outer_init < 4; ++dy_inner_outer_init) {
      C[(((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (dx_inner_outer_init * 16384)) + (dy_inner_outer_init * 16))] = 0.000000e+00f;
    }
  }
  
  for (int k_outer = 0; k_outer < 64; ++k_outer) {
    __syncthreads();
    
    for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
      A_shared[(((((int)threadIdx.x) * 16) + ((int)threadIdx.y)) + (ax0_outer * 256))] = A[((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * 1024) + ((int)threadIdx.y)) + (k_outer * 16)) + (ax0_outer * 16384))];
    }

    for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
      B_shared[(((((int)threadIdx.x) * 64) + ((int)threadIdx.y)) + (ax1_outer * 16))] = B[(((((((int)blockIdx.y) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (k_outer * 16384)) + (ax1_outer * 16))];
    }
    __syncthreads();
    
    for (int k_inner = 0; k_inner < 16; ++k_inner) {
      for (int dx_inner_outer = 0; dx_inner_outer < 4; ++dx_inner_outer) {
        for (int dy_inner_outer = 0; dy_inner_outer < 4; ++dy_inner_outer) {
          C[(((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (dx_inner_outer * 16384)) + (dy_inner_outer * 16))] = (C[(((((((((int)blockIdx.x) * 1024) + ((int)blockIdx.y)) + (((int)threadIdx.x) * 16)) * 64) + ((int)threadIdx.y)) + (dx_inner_outer * 16384)) + (dy_inner_outer * 16))] + (A_shared[(((((int)threadIdx.x) * 16) + k_inner) + (dx_inner_outer * 256))] * B_shared[((((int)threadIdx.y) + (k_inner * 64)) + (dy_inner_outer * 16))]));
        }
      }
    }
  }
}

__global__ void threeMatrixMulV3(int n, const float *A, const float *B,
									const float *C, const float *D, const float *T)
{
	grid_group grid = this_grid();

	twoMatrixMul(n, B, C, T);

	grid.sync();

	twoMatrixMul(n, A, T, D);
}

int main(int argc, char const *argv[])
{
	cublasStatus_t status;
	cublasHanlde_t hanlde;

	int dev_id = findCudaDevice(argc, (const char **)arv);
	cudaDeviceProp device_prop;
	checkCudaErrors(cudaGetDeviceProperties(&device_prop, device_id));

	if ((device_prop.major << 4) + device_prop.minor < 0x35)
	{
		printf("Device API is not supported when CC <= 3.5\n");
		goto finish;
	}

	float *HA, *HB, *HC, *HD;
	allocate_host_memory(HA, SIZE, finish);
	allocate_host_memory(HB, SIZE, finish);
	allocate_host_memory(HC, SIZE, finish);
	allocate_host_memory(HD, SIZE, finish);

	random_fill(HA, n);
	random_fill(HB, n);
	random_fill(HC, n);

	float *DA, *DB, *DC, *DD;
	allocate_device_memory(DA, SIZE, finish);
	allocate_device_memory(DB, SIZE, finish);
	allocate_device_memory(DC, SIZE, finish);
	allocate_device_memory(DD, SIZE, finish);

	initialize_device_matrices(HA, DA, SIZE, finish);
	initialize_device_matrices(HB, DB, SIZE, finish);
	initialize_device_matrices(HC, DC, SIZE, finish);
	initialize_device_matrices(HD, DD, SIZE, finish);

	cublasStatus_t *dev_status;
	allocate_device_memory(dev_status, 1, finish);		

	threeMatrixMulCuda8<<<1, 1>>>(dev_status, N, DA, DB, DC, DD);

	cudaError_t error;
	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		fprintf(stderr, "cuda kernel execution failed: %s\n",
			cudaGetErrorString(error));
		goto finish;
	}

	if ((error = cudaMemCpy(&status, dev_status, sizeof(cublasStatus_t),
		cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		fprintf(stderr, "Device to host memory copy failed: %s\n",
			cudaGetLastError(error));
		goto finish;
	}

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUBLAS device API call failed: %d\n", status);
		goto finish;
	}

	if ((error = cudaMemCpy(&HD, DD, sizeof(float) * SIZE,
		cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		fprintf(stderr, "Device to host memory copy failed: %s\n",
			cudaGetLastError(error));
		goto finish;
	}

	free_device_memory(dev_status, finish);
	free_device_memory(DA, finish);
	free_device_memory(DB, finish);
	free_device_memory(DC, finish);
	free_device_memory(DD, finish);

	return 0;
}