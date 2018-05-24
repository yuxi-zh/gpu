
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

#define ErrorCase(name) \
    case name: \
        return # name;

const char *cudriverGetErrorString(CUresult error)
{
    switch (error)
    {
        ErrorCase(CUDA_SUCCESS);
        ErrorCase(CUDA_ERROR_INVALID_VALUE);
        ErrorCase(CUDA_ERROR_OUT_OF_MEMORY);
        ErrorCase(CUDA_ERROR_NOT_INITIALIZED);
        ErrorCase(CUDA_ERROR_DEINITIALIZED);
        ErrorCase(CUDA_ERROR_PROFILER_DISABLED);
        ErrorCase(CUDA_ERROR_PROFILER_NOT_INITIALIZED);
        ErrorCase(CUDA_ERROR_PROFILER_ALREADY_STARTED);
        ErrorCase(CUDA_ERROR_PROFILER_ALREADY_STOPPED);
        ErrorCase(CUDA_ERROR_NO_DEVICE);
        ErrorCase(CUDA_ERROR_INVALID_DEVICE);
        ErrorCase(CUDA_ERROR_INVALID_IMAGE);
        ErrorCase(CUDA_ERROR_INVALID_CONTEXT);
        ErrorCase(CUDA_ERROR_CONTEXT_ALREADY_CURRENT);
        ErrorCase(CUDA_ERROR_MAP_FAILED);
        ErrorCase(CUDA_ERROR_UNMAP_FAILED);
        ErrorCase(CUDA_ERROR_ARRAY_IS_MAPPED);
        ErrorCase(CUDA_ERROR_ALREADY_MAPPED);
        ErrorCase(CUDA_ERROR_NO_BINARY_FOR_GPU);
        ErrorCase(CUDA_ERROR_ALREADY_ACQUIRED);
        ErrorCase(CUDA_ERROR_NOT_MAPPED);
        ErrorCase(CUDA_ERROR_NOT_MAPPED_AS_ARRAY);
        ErrorCase(CUDA_ERROR_NOT_MAPPED_AS_POINTER);
        ErrorCase(CUDA_ERROR_ECC_UNCORRECTABLE);
        ErrorCase(CUDA_ERROR_UNSUPPORTED_LIMIT);
        ErrorCase(CUDA_ERROR_CONTEXT_ALREADY_IN_USE);
        ErrorCase(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED);
        ErrorCase(CUDA_ERROR_INVALID_PTX);
        ErrorCase(CUDA_ERROR_INVALID_GRAPHICS_CONTEXT);
        ErrorCase(CUDA_ERROR_NVLINK_UNCORRECTABLE);
        ErrorCase(CUDA_ERROR_JIT_COMPILER_NOT_FOUND);
        ErrorCase(CUDA_ERROR_INVALID_SOURCE);
        ErrorCase(CUDA_ERROR_FILE_NOT_FOUND);
        ErrorCase(CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND);
        ErrorCase(CUDA_ERROR_SHARED_OBJECT_INIT_FAILED);
        ErrorCase(CUDA_ERROR_OPERATING_SYSTEM);
        ErrorCase(CUDA_ERROR_INVALID_HANDLE);
        ErrorCase(CUDA_ERROR_NOT_FOUND);
        ErrorCase(CUDA_ERROR_NOT_READY);
        ErrorCase(CUDA_ERROR_ILLEGAL_ADDRESS);
        ErrorCase(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
        ErrorCase(CUDA_ERROR_LAUNCH_TIMEOUT);
        ErrorCase(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING);
        ErrorCase(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED);
        ErrorCase(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED);
        ErrorCase(CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE);
        ErrorCase(CUDA_ERROR_CONTEXT_IS_DESTROYED);
        ErrorCase(CUDA_ERROR_ASSERT);
        ErrorCase(CUDA_ERROR_TOO_MANY_PEERS);
        ErrorCase(CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED);
        ErrorCase(CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED);
        ErrorCase(CUDA_ERROR_HARDWARE_STACK_ERROR);
        ErrorCase(CUDA_ERROR_ILLEGAL_INSTRUCTION);
        ErrorCase(CUDA_ERROR_MISALIGNED_ADDRESS);
        ErrorCase(CUDA_ERROR_INVALID_ADDRESS_SPACE);
        ErrorCase(CUDA_ERROR_INVALID_PC);
        ErrorCase(CUDA_ERROR_LAUNCH_FAILED);
        ErrorCase(CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE);
        ErrorCase(CUDA_ERROR_NOT_PERMITTED);
        ErrorCase(CUDA_ERROR_NOT_SUPPORTED);
        ErrorCase(CUDA_ERROR_UNKNOWN);
    }
    return "<unknown>";
}

#define CHECK_CUDA_CALL(call) \
	__CHECK_CALL__(CUDA, call, cudaError_t, cudaSuccess, cudaGetErrorString)

#define CHECK_CUBLAS_CALL(call) \
	__CHECK_CALL__(CUBLAS, call, cublasStatus_t, CUBLAS_STATUS_SUCCESS, cublasGetErrorString)

#define CHECK_CUDNN_CALL(call) \
    __CHECK_CALL__(CUDNN, call, cudnnStatus_t, CUDNN_STATUS_SUCCESS, cudnnGetErrorString)

#define CHECK_CUDADriver_CALL(call) \
    __CHECK_CALL__(CUDADriver, call, CUresult, CUDA_SUCCESS, cudriverGetErrorString)

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
