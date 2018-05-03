#include "utils.h"
#include "tensor.h"
#include <vector>

// // input shape = [batch, channel, i_height, i_width]
// // filter shape = [f_height, f_width, channel, number]
// // stride = [h_stride, w_stride]
// // output shape = [batch, channel * f_height * f_width, 
// // ceil_div(i_height - f_height, h_stride) * ceil_div(i_width - f_width, w_stride)]
// __global__ void im2col(Tensor<4> *input, Tensor<4> *filter, Tensor<3> *output, int *stride) {

// 	int o_height = ceil_div(input->shape[2] - filter->shape[0], stride[0]);
// 	int o_width = ceil_div(input->shape[3] - filter->shape[1], stride[1]);
// 	int batch = blockIdx.z / input->shape[1];
// 	int channel = blockIdx.z % input->shape[1];
	
// 	for (int i2 = 0; i2 < filter->shape[0] * filter->shape[1]; i2++) {
// 		int y = channel * filter->shape[0] * filter->shape[1] + i2;
// 		int i0 = blockIdx.y * blockDim.y + threadIdx.y;
// 		int i1 = blockIdx.x * blockDim.x + threadIdx.x;
// 		int x = i0 * i1;
// 		int yy = i0 * stride[0] + i2 / filter->shape[1];
// 		int xx = i1 * stride[0] + i2 % filter->shape[1];
// 		float value;
// 		if (i0 < o_height && i1 < o_width) {
// 			int input_index[4] = {batch, channel, yy, xx};
// 			int output_index[3] = {batch, y, x};
// 			input->GetElement(input_index, &value);
// 			output->SetElement(output_index, value);
// 		}
// 	}
// }

// extern __shared__ float shared[];

// __global__ void im2col_v1(Tensor<4> *input, Tensor<4> *filter, Tensor<3> *output, int *stride) {

// 	int o_height = ceil_div(input->shape[2] - filter->shape[0], stride[0]);
// 	int o_width = ceil_div(input->shape[3] - filter->shape[1], stride[1]);
// 	int batch = blockIdx.z / input->shape[1];
// 	int channel = blockIdx.z % input->shape[1];
	
// 	int shared_height = (blockDim.y - 1) * stride[0] + filter->shape[0];
// 	int shared_width = (blockDim.x - 1) * stride[1] + filter->shape[1];

// 	int i0 = blockIdx.y * blockDim.y;
// 	int i1 = blockIdx.x * blockDim.x;

// 	int _i0 = i0 + threadIdx.y;
// 	int _i1 = i0 + threadIdx.x;

// 	for (int sh = 0; sh < ceil_div(shared_height, blockDim.y); sh++) {
// 		for (int sw = 0; sw < ceil_div(shared_width, blockDim.x); sw++) {
// 			int _sh_ = sh * blockDim.y + threadIdx.y;
// 			int _sw_ = sw * blockDim.x + threadIdx.x;
// 			int index = _sh_ * shared_width + _sw_;
// 			int yy = i0 * stride[0] + _sh_;
// 			int xx = i1 * stride[1] + _sw_;
// 			if (yy < shared_height && xx < shared_width) {
// 				int input_index[4] = {batch, channel, yy, xx};
// 				input->GetElement(input_index, &shared[index]);
// 			}
// 		}
// 	}
	
// 	__syncthreads();

// 	for (int i2 = 0; i2 < filter->shape[0] * filter->shape[1]; i2++) {
// 		int y = channel * filter->shape[0] * filter->shape[1] + i2;
// 		int x = _i0 * _i1;

// 		int yy = threadIdx.y * stride[0] + i2 / filter->shape[1];
// 		int xx = threadIdx.x * stride[1] + i2 % filter->shape[1];
// 		int index = yy * shared_width + xx;

// 		if (_i0 < o_height && _i1 < o_width) {
// 			int output_index[3] = {batch, y, x};
// 			output->SetElement(output_index, shared[index]);
// 		}
// 	}
// }

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(int batch, const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  timing([&] { 
  		for (int _ = 0; _ < batch; _++) {
  		im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);} });
}

using namespace std;

void drive_im2col(vector<int> input_shape, vector<int> filter_shape, 
	vector<int> stride, vector<int> padding, int thread) {
  // int height_col = (height + 2 * pad_h -
  //     (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  // int width_col = (width + 2 * pad_w -
  //     (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	int conv_height = (input_shape[2] - filter_shape[0] + 2 * padding[0]) / stride[0] + 1;
	int conv_width = (input_shape[3] - filter_shape[1] + 2 * padding[1]) /  stride[1] + 1;
	int output_shape[3] = { input_shape[0], 
		input_shape[1] * filter_shape[0] * filter_shape[1], conv_height * conv_width };

	// Tensor<4>* input = make_device_tensor<4>(input_shape.data());
	// Tensor<4>* filter = make_device_tensor<4>(filter_shape.data());
	// Tensor<3>* output = make_device_tensor<3>(output_shape);

	// dim3 grid(ceil_div(conv_width, thread), ceil_div(conv_height, thread), 
	// 	input_shape[0] * input_shape[1]);
	// dim3 block(thread, thread);
	// int shared_size = ((thread - 1) * stride[0] + filter_shape[0]) * ((thread - 1) * stride[1] + filter_shape[1]) * sizeof(float);

	// int *device_stride = nullptr;
	// CHECK_CUDA_CALL(cudaMalloc((void **)&device_stride, sizeof(int) * 2));
	// CHECK_CUDA_CALL(cudaMemcpy(device_stride, stride.data(), sizeof(int) * 2, cudaMemcpyHostToDevice));

	// timing([&] { im2col<<<grid, block>>>(input, filter, output, device_stride); });
	// timing([&] { im2col_v1<<<grid, block, shared_size>>>(input, filter, output, device_stride); });

	float* input_data = nullptr;
	float* output_data = nullptr;
	int input_size = input_shape[1] * input_shape[2] * input_shape[3];
	int output_size = output_shape[1] * output_shape[2];
	CHECK_CUDA_CALL(cudaMalloc((void **)&input_data, sizeof(float) * input_size));
	CHECK_CUDA_CALL(cudaMalloc((void **)&output_data, sizeof(float) * output_size));
	im2col_gpu<float>(input_shape[0], input_data, input_shape[1], input_shape[2], input_shape[3],
		filter_shape[0], filter_shape[1], padding[0], padding[1], stride[0], stride[1], 1, 1, output_data);

	cudaFree(input_data);
	cudaFree(output_data);
}
