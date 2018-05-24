#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "utils.h"

template <int NDim, typename DType = float>
class Tensor {
public:
	DType *data;
	int shape[NDim];

	__device__
	void GetElement(int index[NDim], DType *value) {
		int _index = 0;
		#pragma unroll
		for (int i = 0; i < NDim; i++) {
			_index = _index * shape[i] + index[i];
		}
		*value = data[_index];		
	}

	__device__ 
	void SetElement(int index[NDim], DType value) {
		int _index = 0;
		#pragma unroll
		for (int i = 0; i < NDim; i++) {
			_index = _index * shape[i] + index[i];
		}
		data[_index] = value;		
	}
};

template<int NDim, typename DType=float>
Tensor<NDim, DType>* make_device_tensor(int shape[NDim]) {

	Tensor<NDim, DType> host_tensor;
	memcpy(host_tensor.shape, shape, NDim * sizeof(int));
	DType *device_data = nullptr;
	int size = 1;
	for (int i = 0; i < NDim; i++) size *= shape[i];
	CHECK_CUDA_CALL(cudaMalloc((void **)&device_data, sizeof(DType) * size));
	host_tensor.data = device_data;

	Tensor<NDim, DType>* device_tensor = nullptr;
	CHECK_CUDA_CALL(cudaMalloc((void **)&device_tensor, sizeof(Tensor<NDim, DType>)));
	CHECK_CUDA_CALL(cudaMemcpy(device_tensor, &host_tensor, 
		sizeof(Tensor<NDim, DType>), cudaMemcpyHostToDevice));

	return device_tensor;
}

#endif