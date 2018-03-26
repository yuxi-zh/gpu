#include "utils.h"
#include "stdio.h"
#include "stdlib.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"


inline void random_fill(float *A, int size) {
	for (int i = 0; i < size; ++i)
		A[i] = rand() / (float)RAND_MAX;
}

void time_evaluator(int m, int n, int l) {

	cublasHandle_t handle;
	float *T= NULL, alpha = 1.0f, beta = 0.0f;
	
	CHECK_CUBLAS_CALL(cublasCreate(&handle));
	
	float *HA, *HB, *HC, *HD, *TD;
	ASSERT((HA = (float *)malloc(sizeof(float) * (l * n))) != NULL);
	ASSERT((HB = (float *)malloc(sizeof(float) * (l * m))) != NULL);
	ASSERT((HC = (float *)malloc(sizeof(float) * (m * n))) != NULL);
	
	random_fill(HA, (l * n));
	random_fill(HB, (l * m));

	float *DA, *DB, *DC, *DD;
	CHECK_CUDA_CALL(cudaMalloc((void **)&DA, sizeof(float) * (l * n)));
	CHECK_CUDA_CALL(cudaMalloc((void **)&DB, sizeof(float) * (l * m)));
	CHECK_CUDA_CALL(cudaMalloc((void **)&DC, sizeof(float) * (m * n)));

	CHECK_CUBLAS_CALL(cublasSetVector((l * n), sizeof(float), HA, 1, DA, 1));
	CHECK_CUBLAS_CALL(cublasSetVector((l * m), sizeof(float), HB, 1, DB, 1));
	
	CHECK_CUBLAS_CALL(cublasCreate(&hanlde));
	CHECK_CUDA_CALL(cudaMalloc((void **)&T, sizeof(float) * n * n));

	cudaEvent_t start, end;
	float sum = 0.0, tmp;
	int num_runs = 10;
	for (int _ = 0; _ < num_runs; _++)
	{
		CHECK_CUDA_CALL(cudaEventCreate(&start));
		CHECK_CUDA_CALL(cudaEventCreate(&end));
		CHECK_CUDA_CALL(cudaEventRecord(start, 0));
		CHECK_CUBLAS_CALL(cublasSgemm(
			hanlde, CUBLAS_OP_T, CUBLAS_OP_N, m, n, l, &alpha, B, l, A, l, &beta, C, m
		));
		CHECK_CUDA_CALL(cudaEventRecord(end, 0));
		CHECK_CUDA_CALL(cudaEventSynchronize(end));
		CHECK_CUDA_CALL(cudaEventElapsedTime(&tmp, start, end));
		CHECK_CUDA_CALL(cudaEventDestroy(start));
		CHECK_CUDA_CALL(cudaEventDestroy(end));
		sum += tmp;
	}
	float t = sum / num_runs;
	float GFLOPS = (2 * m * n * l) / (t * 1e3) / 1e6;
	print("(m, n, l) = (%4d, %4d, %4d), average time cost of %d runs = %g ms, %g GFLOPS." % (m, n, l, num_runs, t * 1e3, GFLOPS));

	CHECK_CUBLAS_CALL(cublasDestroy(hanlde));

	CHECK_CUDA_CALL(cudaFree(DA));
	CHECK_CUDA_CALL(cudaFree(DB));
	CHECK_CUDA_CALL(cudaFree(DC));	
	
	free(HA);
	free(HB);
	free(HC);
}

int main(int argc, char const *argv[])
{
	int n_device = 0;
	CHECK_CUDA_CALL(cudaGetDeviceCount(&n_device));
	ASSERT_MSG(n_device == 1, "Only consider one device case");

	cudaDeviceProp device_prop;
	CHECK_CUDA_CALL(cudaGetDeviceProperties(&device_prop, 0));
	ASSERT_MSG((device_prop.major << 4) + device_prop.minor >= 0x35,
		"Device API is not supported when cc <= 3.5");

	std::vector<int> scale = {64, 128, 256, 512, 1024, 2048};
	std::vector<int> n = {1, 2, 4, 18, 16, 32, 64};

	for (int i : scale) {
		time_evaluator(scale, scale, scale);
	}

	for (int i ： n) {
		time_evaluator(512, i, 512);
	}


	return 0;
}