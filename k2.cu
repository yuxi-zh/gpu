#include "stdio.h"
#include "stdlib.h"
#include "errno.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "helper_cuda.h"

__global__ void threeMatrixMulCuda8(cublasStatus_t *returnValue, int n,
									const float *A, const float *B,
									const float *C, const float *D)
{
	cublasHanlde_t hanlde;
	cublasStatus_t status = cublasCreate(&hanlde);

	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;

	float *T = (float *)malloc(sizeof(float) * n * n);
	if (T == nullptr)
		goto finish;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 1.0f, B, n, C, n, 0.0f, D, n
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