
#include "utils.h"
#include "cuda_runtime.h"

#define check_attr(value, attr, device) \
	CHECK_CUDA_CALL(cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerBlock, device)); \
	printf("%s\t\t= %d\n", #attr, value);

int main(int argc, char const *argv[])
{
	int value;

	check_attr(value, cudaDevAttrMultiProcessorCount, 0);

	check_attr(value, cudaDevAttrMaxThreadsPerBlock, 0);
	check_attr(value, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

	check_attr(value, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	check_attr(value, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);

	check_attr(value, cudaDevAttrMaxRegistersPerBlock, 0);
	check_attr(value, cudaDevAttrMaxRegistersPerMultiprocessor, 0);

	return 0;
}