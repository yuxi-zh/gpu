
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>

#define __CHECK_CALL__(name, call, type, success, err2str) do { \
	type error; \
	if ((error = (call)) != success) { \
		fprintf(stderr, "[%s-%d] CHECK_%s_CALL(%s) failed for %s\n", \
			__FUNCTION__, __LINE__, #name, #call, err2str(error) \
		); \
		exit(0); \
	} \
} while(0)

#define ASSERT_MSG(stmt, msg) do { \
	if (!(stmt)) { \
		fprintf(stderr, "[%s-%d] ASSERT(%s) is failed", \
			__FUNCTION__, __LINE__, #stmt\
		); \
		if (strlen(msg) != 0) \
			fprintf(stderr, ", %s", msg); \
		fprintf(stderr, "\n"); \
		exit(0); \
	} \
} while(0)

#define ASSERT(stmt) ASSERT_MSG(stmt, "")

const char *cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define CHECK_CUDA_CALL(call) \
	__CHECK_CALL__(CUDA, call, cudaError_t, cudaSuccess, cudaGetErrorString)

#define CHECK_CUBLAS_CALL(call) \
	__CHECK_CALL__(CUBLAS, call, cublasStatus_t, CUBLAS_STATUS_SUCCESS, cublasGetErrorString)

#define CHECK_CUDNN_CALL(call) \
    __CHECK_CALL__(CUDNN, call, cudnnStatus_t, CUDNN_STATUS_SUCCESS, cudnnGetErrorString)


// void timing(std::function<void(void)> target)
// {
//     cudaEvent_t start, end;
//     float sum = 0.0, tmp;
//     int num_runs = 100;
//     for (int _ = 0; _ < num_runs; _++)
//     {
//         CHECK_CUDA_CALL(cudaEventCreate(&start));
//         CHECK_CUDA_CALL(cudaEventCreate(&end));
//         CHECK_CUDA_CALL(cudaEventRecord(start, 0));
//         target();
//         CHECK_CUDA_CALL(cudaGetLastError());
//         CHECK_CUDA_CALL(cudaEventRecord(end, 0));
//         CHECK_CUDA_CALL(cudaEventSynchronize(end));
//         CHECK_CUDA_CALL(cudaEventElapsedTime(&tmp, start, end));
//         CHECK_CUDA_CALL(cudaEventDestroy(start));
//         CHECK_CUDA_CALL(cudaEventDestroy(end));
//         sum += tmp;
//     }
//     float t = sum / num_runs;
//     printf("%g\n", t);
// }

// #define ceil_div(a, b) (((a) + (b) - 1) / (b))

#endif
