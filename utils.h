
#ifndef __UTILS_H__
#define __UTILS_H__

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

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
			__FUNCTION__, __LINE__, #stmt,\
		); \
		if (strlen(msg) != 0) \
			fprintf(stderr, ", %s", msg); \
		fprintf(stderr, "\n"); \
		exit(0); \
	} \
} while(0)

#define ASSERT(stmt) ASSERT_MSG(stmt, "")

static const char *cublasGetErrorString(cublasStatus_t error)
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

#endif