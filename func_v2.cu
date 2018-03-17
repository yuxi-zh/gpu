
#include "utils.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

__global__
void __func_v2__(cublasStatus_t *status, int n, const float* A, const float* B, 
					const float* C, float* D)
{
	cublasHandle_t hanlde;	
	float *T = NULL, alpha = 1.0f, beta = 0.0f;

	*status = cublasCreate(&hanlde);
	if (*status != CUBLAS_STATUS_SUCCESS)
		return;

	T = (float *)malloc(sizeof(float) * n * n);
	if (T == NULL) {
		*status = CUBLAS_STATUS_ALLOC_FAILED;
		return;
	}

	*status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, C, n, &beta, T, n
	);
	if (*status != CUBLAS_STATUS_SUCCESS)
		return;

	*status = cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, T, n, &beta, D, n
	);
	if (*status != CUBLAS_STATUS_SUCCESS)
		return;

	free(T);
	*status = cublasDestroy(hanlde);
}

void func_v2(int n, const float* A, const float* B, const float* C, float* D)
{
	cublasHandle_t handle;
	CHECK_CUBLAS_CALL(cublasCreate(&handle));

	cublasStatus_t status;
	cublasStatus_t *dev_status;
	CHECK_CUDA_CALL(cudaMalloc((void **)&dev_status, sizeof(cublasStatus_t)));

	__func_v2__<<<1, 1>>>(dev_status, n, A, B, C, D);
	CHECK_CUDA_CALL(cudaGetLastError());

	CHECK_CUDA_CALL(cudaMemcpy(&status, dev_status, sizeof(cublasStatus_t), 
		cudaMemcpyDeviceToHost));
//	ASSERT_MSG(status == CUBLAS_STATUS_SUCCESS, "CUBLAS device API call failed");
	CHECK_CUBLAS_CALL(status);
	CHECK_CUDA_CALL(cudaFree(dev_status));
	CHECK_CUBLAS_CALL(cublasDestroy(handle));
}
