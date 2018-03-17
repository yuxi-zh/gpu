#include "stdio.h"
#include "stdlib.h"
#include "errno.h"
#include "string.h"

#include "cuda_runtime.h"
#include "cublas_v2.h"
// #include "helper_cuda.h"

#define N 			(1024)
#define SIZE 		(N * N)
#define OUTER_RUNS	(50)

#define allocate_host_memory(name, size, errhanlde) 							\
	name = (float *)malloc(sizeof(float) * size); 						\
	if (name == NULL) 														\
	{																			\
		fprintf(stderr, "Host memory allocation for %s failed: %s\n",			\
			#name, strerror(errno));											\
		goto errhanlde;															\
	}

#define allocate_device_memory(name, size, errhanlde)							\
	if (cudaMalloc((void **)&name, size * sizeof(name[0])) != cudaSuccess)		\
	{																			\
		fprintf(stderr, "Device memory allocation for %s failed\n", #name);		\
		goto errhanlde;															\
	}

#define initialize_device_matrices(host, device, size, errhanlde)				\
	status = cublasSetVector(size, sizeof(host[0]), host, 1, device, 1);		\
	if (status != CUBLAS_STATUS_SUCCESS)										\
	{																			\
		fprintf(stderr, "device access from %s to %s error\n", #host, #device);	\
		goto errhanlde;															\
	}

#define free_device_memory(name, errhanlde)										\
	if (cudaFree(name) != cudaSuccess) {								\
		fprintf(stderr, "Memory free %s failed\n", #name);						\
		goto errhanlde;															\
	}

#define random_fill(M, n)														\
	for (int i = 0; i < n; ++i)													\
		M[i] = rand() / (float)RAND_MAX;										\

void threeMatrixMulV0(int n, const float* A, const float* B,
					 const float* C, float* D)
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

void threeMatrixMulV1(int n, const float* A, const float* B, 
					 const float* C, float* D)
{
	cublasHandle_t hanlde;
	cublasStatus_t status = cublasCreate(&hanlde);
	float *T;
	float alpha = 1.0f, beta = 0.0f;

	if (status != CUBLAS_STATUS_SUCCESS)
		goto finish;

	allocate_device_memory(T, SIZE, clean_handle);

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, C, n, &beta, T, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto clean_t;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, T, n, &beta, D, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto clean_t;

clean_t:
	free_device_memory(T, clean_handle);

clean_handle:
	cublasDestroy(hanlde);

finish: return;
}

__global__ void __threeMatrixMulV2(cublasStatus_t *returnValue, int n,
								 const float* A, const float* B,
								 const float* C, float* D)
{
	cublasHandle_t hanlde;
	cublasStatus_t status = cublasCreate(&hanlde);
	float *T;
	float alpha = 1.0f, beta = 0.0f;

	if (status != CUBLAS_STATUS_SUCCESS)
		goto clean_handle;

	T = (float *)malloc(sizeof(float) * SIZE);
	if (T == NULL)
		goto clean_handle;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, C, n, &beta, T, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto clean_t;

	status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, T, n, &beta, D, n
	);
	if (status != CUBLAS_STATUS_SUCCESS)
		goto clean_t;

clean_t:
	free(T);

clean_handle:
	cublasDestroy(hanlde);

	*returnValue = status;
}

void threeMatrixMulV2(int n, const float* A, const float* B, 
						const float* C, float* D)
{
	cublasStatus_t status;
	cublasHandle_t handle;

	status = cublasCreate(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUBLAS initialization failed\n");
		goto finish;
	}

	cublasStatus_t *dev_status;
	allocate_device_memory(dev_status, 1, cds);		

	__threeMatrixMulV2<<<1, 1>>>(dev_status, N, A, B, C, D);

	cudaError_t error;
	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		fprintf(stderr, "cuda kernel execution failed: %s\n",
			cudaGetErrorString(error));
		goto cds;
	}

	if ((error = cudaMemcpy(&status, dev_status, sizeof(cublasStatus_t),
		cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		fprintf(stderr, "Device to host memory copy failed: %s\n",
			cudaGetErrorString(error));
		goto cds;
	}

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "CUBLAS device API call failed: %d\n", status);
		goto cds;
	}

cds:free_device_memory(dev_status, chandle);

chandle:cublasDestroy(handle);

finish:return;

}

typedef void (*threeMatrixMulFunc)(int, const float*, const float*, 
									const float*, float*);

void __evaluate(threeMatrixMulFunc threeMatrixMul, int n, const float* A,
			 const float* B, const float* C, float* D)
{
	cudaEvent_t start, end;
	float sum = 0.0, tmp;
	for (int _ = 0; _ < OUTER_RUNS; _++)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);
		threeMatrixMul(n, A, B, C, D);
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&tmp, start, end);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
		sum += tmp * 1000.0;
	}

	printf("%f\n", sum/OUTER_RUNS);
}

#define evaluate(threeMatrixMul, n, A, B, C, D)									\
	printf("Evalute %s : ", #threeMatrixMul);									\
	__evaluate(threeMatrixMul, n, A, B, C, D);										

int main(int argc, char const *argv[])
{
	cublasStatus_t status;
	cublasHandle_t handle;

	int n_device = 0;
	cudaGetDeviceCount(&n_device);

	if (n_device != 1)
	{
		fprintf(stderr, "Device count = %d\n", n_device);
		goto finish;
	}

	cudaDeviceProp device_prop;
	cudaGetDeviceProperties(&device_prop, 0);

	if ((device_prop.major << 4) + device_prop.minor < 0x35)
	{
		printf("Device API is not supported when CC <= 3.5\n");
		goto finish;
	}

	status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        goto chandle;
    }

	float *HA, *HB, *HC, *HD;
	allocate_host_memory(HA, SIZE, finish);
	allocate_host_memory(HB, SIZE, cha);
	allocate_host_memory(HC, SIZE, chb);
	allocate_host_memory(HD, SIZE, chc);

	random_fill(HA, SIZE);
	random_fill(HB, SIZE);
	random_fill(HC, SIZE);

	float *DA, *DB, *DC, *DD;
	allocate_device_memory(DA, SIZE, chd);
	allocate_device_memory(DB, SIZE, cda);
	allocate_device_memory(DC, SIZE, cdb);
	allocate_device_memory(DD, SIZE, cdc);

	initialize_device_matrices(HA, DA, SIZE, cdd);
	initialize_device_matrices(HB, DB, SIZE, cdd);
	initialize_device_matrices(HC, DC, SIZE, cdd);
	initialize_device_matrices(HD, DD, SIZE, cdd);

	evaluate(threeMatrixMulV1, N, DA, DB, DC, DD);
	evaluate(threeMatrixMulV2, N, DA, DB, DC, DD);

cdd:free_device_memory(DD, finish);
cdc:free_device_memory(DC, finish);
cdb:free_device_memory(DB, finish);
cda:free_device_memory(DA, finish);

chd:free(HD);
chc:free(HC);
chb:free(HB);
cha:free(HA);

chandle:cublasDestroy(handle);

finish:return 0;

}