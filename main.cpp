#include "utils.h"
#include "stdio.h"
#include "stdlib.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

#define N 					(1024)
#define SIZE 				(N * N)
#define RAND_MAX			(1e6)
#define OUTER_RUNS			(50)

inline void random_fill(float *A, int size) {
	for (int i = 0; i < size; ++i)
		A[i] = rand() / (float)RAND_MAX;
}

typedef void (*funcType)(int, const float*, const float*, const float*, float*);

void __evaluate__(funcType func,
				int n, const float* A, const float* B, const float* C, float* D)
{
	cudaEvent_t start, end;
	float sum = 0.0, tmp;
	for (int _ = 0; _ < OUTER_RUNS; _++)
	{
		CHECK_CUDA_CALL(cudaEventCreate(&start));
		CHECK_CUDA_CALL(cudaEventCreate(&end));
		CHECK_CUDA_CALL(cudaEventRecord(start, 0));
		func(n, A, B, C, D);
		CHECK_CUDA_CALL(cudaEventRecord(end, 0));
		CHECK_CUDA_CALL(cudaEventSynchronize(end));
		CHECK_CUDA_CALL(cudaEventElapsedTime(&tmp, start, end));
		CHECK_CUDA_CALL(cudaEventDestroy(start));
		CHECK_CUDA_CALL(cudaEventDestroy(end));
		sum += tmp * 1000.0;
	}

	printf("%f\n", sum/OUTER_RUNS);
}

void __close__(int n, const float* v1, const float *v2, float error)
{
	int miss = 0;
	for (int i = 0; i < n; ++i)
		if (fabs(v1[i] - v2[i]) > error) miss += 1;
	printf("%0.2f%\n", miss / (float)n * 100);
}

#define EVALUATE(func, n, A, B, C, D, HD, TD) do { \
	printf("%s :", #func); \
	__evaluate__(func, n, A, B, C, D); \
	CHECK_CUDA_CALL(cudaMemcpy(HD, TD, n, cudaMemcpyDeviceToHost)); \
	__close__(n, HD, TD, 1e5); \
} while(0)

void func_v0(int n, const float* A, const float* B, const float* C, float* D) 
{
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			D[i * n + j] = 0;
			for (int k = 0; k < n; ++k) {
				for (int l = 0; l < n; ++l) {
					D[i * n + j] += A[i *n + k] * B[k * n + l] * C[l * n + j];
				}
			}
		}
	}
}

void func_v1(int n, const float* A, const float* B, const float* C, float* D)
{
	cublasHandle_t hanlde;
	float *T= NULL, alpha = 1.0f, beta = 0.0f;
	
	CHECK_CUBLAS_CALL(cublasCreate(&hanlde));
	CHECK_CUDA_CALL(cudaMalloc((void **)&T, sizeof(float) * n));

	CHECK_CUBLAS_CALL(cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B, n, C, n, &beta, T, n
	));
	CHECK_CUBLAS_CALL(cublasSgemm(
		hanlde, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, T, n, &beta, D, n
	));
	CHECK_CUDA_CALL(cudaFree(T));
	CHECK_CUBLAS_CALL(cublasDestroy(hanlde));
}

extern 
void func_v2(cublasStatus_t*, int, const float*, const float*, const float*, float*);

int main(int argc, char const *argv[]) {

	int n_device = 0;
	CHECK_CUDA_CALL(cudaGetDeviceCount(&n_device));
	ASSERT_MSG(n_device == 1, "Only consider one device case");

	cudaDeviceProp device_prop;
	CHECK_CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));
	ASSERT_MSG((device_prop.major << 4) + device_prop.minor < 0x35,
		"Device API is not supported when cc <= 3.5");
	
	cublasHandle_t handle;
	CHECK_CUBLAS_CALL(cublasCreate(&handle));

	float *HA, *HB, *HC, *HD, *TD;
	ASSERT((HA = (float *)malloc(sizeof(float) * SIZE)) != NULL);
	ASSERT((HB = (float *)malloc(sizeof(float) * SIZE)) != NULL);
	ASSERT((HC = (float *)malloc(sizeof(float) * SIZE)) != NULL);
	ASSERT((HD = (float *)malloc(sizeof(float) * SIZE)) != NULL);
	ASSERT((TD = (float *)malloc(sizeof(float) * SIZE)) != NULL);

	random_fill(HA, SIZE);
	random_fill(HB, SIZE);
	random_fill(HC, SIZE);

	func_v0(N, HA, HB, HC, TD);

	float *DA, *DB, *DC, *DD;
	CHECK_CUDA_CALL(cudaMalloc((void **)&DA, sizeof(float) * SIZE));
	CHECK_CUDA_CALL(cudaMalloc((void **)&DB, sizeof(float) * SIZE));
	CHECK_CUDA_CALL(cudaMalloc((void **)&DC, sizeof(float) * SIZE));
	CHECK_CUDA_CALL(cudaMalloc((void **)&DD, sizeof(float) * SIZE));

	CHECK_CUBLAS_CALL(cublasSetVector(SIZE, sizeof(float), HA, 1, DA, 1));
	CHECK_CUBLAS_CALL(cublasSetVector(SIZE, sizeof(float), HB, 1, DB, 1));
	CHECK_CUBLAS_CALL(cublasSetVector(SIZE, sizeof(float), HC, 1, DC, 1));

	EVALUATE(func_v1, N, DA, DB, DC, DD, HD, TD);
	EVALUATE(func_v2, N, DA, DB, DC, DD, HD, TD);

	CHECK_CUDA_CALL(cudaFree(DA));
	CHECK_CUDA_CALL(cudaFree(DB));
	CHECK_CUDA_CALL(cudaFree(DC));
	CHECK_CUDA_CALL(cudaFree(DD));
	
	free(HA);
	free(HB);
	free(HC);
	free(HD);

	CHECK_CUBLAS_CALL(cublasDestroy(handle));

	return 0;
}