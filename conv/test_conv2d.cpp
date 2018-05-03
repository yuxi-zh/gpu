#include <cuda_runtime.h>

#include "utils.h"




void timing(function<void(void)> target)
{
	cudaEvent_t start, end;
	float sum = 0.0, tmp;
	int num_runs = 100;
	for (int _ = 0; _ < num_runs; _++)
	{
		CHECK_CUDA_CALL(cudaEventCreate(&start));
		CHECK_CUDA_CALL(cudaEventCreate(&end));
		CHECK_CUDA_CALL(cudaEventRecord(start, 0));
		target();
		CHECK_CUDA_CALL(cudaGetLastError());
		CHECK_CUDA_CALL(cudaEventRecord(end, 0));
		CHECK_CUDA_CALL(cudaEventSynchronize(end));
		CHECK_CUDA_CALL(cudaEventElapsedTime(&tmp, start, end));
		CHECK_CUDA_CALL(cudaEventDestroy(start));
		CHECK_CUDA_CALL(cudaEventDestroy(end));
		sum += tmp;
	}
	float t = sum / num_runs;
	printf("%g", t * 1e3);
}

void random(float *a, int n)
{
	for (int i = 0; i < n; i++)
		a[i] = rand() % 100;
}

void test_close(float *a, float *b, int n)
{
	int count = 0;
	for (int i = 0; i < n; i++)
		if (fabs(a[i] - b[i]) > 1e-2)
			count += 1;
	cout << "close rate = " << (static_cast<float>(count) / n) << endl;
}

tuple<int, int, int, int> get_pad_tuple(int a_h, int a_w,
										int w_h, int w_w,
										int s_h, int s_w)
{
	int o_h, o_w, p_h, p_w;

 	o_h = ceildiv(a_h, s_h);
 	o_w = ceildiv(a_w, s_w);
 	p_h = ceildiv((o_h - 1) * s_h + w_h - a_h, 2);
 	p_w = ceildiv((o_w - 1) * s_w + w_w - a_w, 2);
}

void test_cudnn_conv2d(int a_n, int a_c, int a_h, int a_w,
					   int w_c, int w_f, int w_h, int w_w,
					   int s_h, int s_w, 
					   string padding, string layout)
{
	float* ad = nullptr;
	float* wd = nullptr;
	float* od = nullptr;
	float* ha = nullptr;
	float* hw = nullptr;
	float* ho = nullptr;

	int o_h, o_w, p_h, p_w;
	tie(o_h, o_w, p_h, p_w) = get_pad_tuple(a_h, a_w, w_h, w_w, s_h, s_w);

	cublasHandle_t handle;
	CHECK_CUBLAS_CALL(cublasCreate(&handle));

	ASSERT((ha = (float *)malloc(sizeof(float) * a_n * a_h * a_w * a_c)) != NULL);
	ASSERT((hw = (float *)malloc(sizeof(float) * w_h * w_w * w_c * w_f)) != NULL);
	ASSERT((ho = (float *)malloc(sizeof(float) * a_n * o_h * o_w * w_f)) != NULL);

	CHECK_CUDA_CALL(cudaMalloc((void **)&ad, sizeof(float) * a_n * a_h * a_w * a_c));
	CHECK_CUDA_CALL(cudaMalloc((void **)&wd, sizeof(float) * w_h * w_w * w_c * w_f));
	CHECK_CUDA_CALL(cudaMalloc((void **)&od, sizeof(float) * a_n * o_h * o_w * w_f));

	random(ha, a_n * a_h * a_w * a_c);
	random(hw, w_h * w_w * w_c * w_f);

	CHECK_CUBLAS_CALL(cublasSetVector(a_n * a_h * a_w * a_c, sizeof(float), ha, 1, ad, 1));
	CHECK_CUBLAS_CALL(cublasSetVector(w_h * w_w * w_c * w_f, sizeof(float), hw, 1, wd, 1));

	cudnnHandle_t cudnn;
	CHECK_CUDNN_CALL(cudnnCreate(&cudnn));

	cudnnTensorDescriptor_t adesc;
	CHECK_CUDNN_CALL(cudnnCreateTensorDescriptor(&adesc));
	CHECK_CUDNN_CALL(cudnnSetTensor4dDescriptor(
		adesc, /*format=*/CUDNN_TENSOR_NHWC,/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/a_n, /*channels=*/a_c,));

	cudnnTensorDescriptor_t odesc;
	CHECK_CUDNN_CALL(cudnnCreateTensorDescriptor(&odesc));
	CHECK_CUDNN_CALL(cudnnSetTensor4dDescriptor(
		odesc, /*format=*/CUDNN_TENSOR_HWCN,/*dataType=*/CUDNN_DATA_FLOAT,
		/*batch_size=*/a_n, /*channels=*/w_f));

	cudnnFilterDescriptor_t wdesc;
	CHECK_CUDNN_CALL(cudnnCreateFilterDescriptor(&wdesc));
	CHECK_CUDNN_CALL(cudnnSetFilter4dDescriptor(
		wdesc,/*dataType=*/CUDNN_DATA_FLOAT,/*format=*/CUDNN_TENSOR_NHWC,
		/*out_channels=*/w_f, /*in_channels=*/w_c, 
		/*kernel_height=*/w_h, /*kernel_width=*/w_w));

	cudnnConvolutionDescriptor_t cdesc;
	CHECK_CUDNN_CALL(cudnnCreateConvolutionDescriptor(&cdesc));
	CHECK_CUDNN_CALL(cudnnSetConvolution2dDescriptor(
		cdesc, /*pad_height=*/1, /*pad_width=*/1, /*vertical_stride=*/s_h,
		/*horizontal_stride=*/s_w, /*dilation_height=*/1, /*dilation_width=*/1,
		/*mode=*/CUDNN_CROSS_CORRELATION, /*computeType=*/CUDNN_DATA_FLOAT));	

	cudnnConvolutionFwdAlgo_t calgo;
	CHECK_CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn,
		adesc, wdesc, cdesc, odesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		/*memoryLimitInBytes=*/0, &calgo));

	size_t workspace_bytes = 0;
	CHECK_CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
		adesc, wdesc, cdesc, odesc, calgo, &workspace_bytes));

	float *sd = nullptr
	CHECK_CUDA_CALL(cudaMalloc((void **)&sd, workspace_bytes));

	const float alpha = 1, beta = 0;
	timing([&] { CHECK_CUDNN_CALL(cudnnConvolutionForward(cudnn,
		&alpha, adesc, ad, wdesc, wd, cdesc, calgo, sd, workspace_bytes, 
		&beta, odesc, _od)); });

	CHECK_CUDNN_CALL(cudnnDestroyTensorDescriptor(adesc));
	CHECK_CUDNN_CALL(cudnnDestroyTensorDescriptor(odesc));
	CHECK_CUDNN_CALL(cudnnDestroyFilterDescriptor(wdesc));
	CHECK_CUDNN_CALL(cudnnDestroyConvolutionDescriptor(cdesc));
	CHECK_CUDNN_CALL(cudnnDestroy(cudnn));
	CHECK_CUBLAS_CALL(cublasDestroy(handle));
	CHECK_CUDA_CALL(cudaFree(sd));
	CHECK_CUDA_CALL(cudaFree(ad));
	CHECK_CUDA_CALL(cudaFree(wd));
	CHECK_CUDA_CALL(cudaFree(od));
	free(ha);
	free(hw);
	free(ho);

	return 0;
}

string tvm_conv2d = 
"import os\n"
"import tvm\n"
"import topi\n"
"import argparse\n"
"from topi.util import get_const_tuple\n"
"parser = argparse.ArgumentParser(description='Generate device function of conv2d_hwcn')\n"
"parser.add_argument('batch', type=int)\n"
"parser.add_argument('in_channel', type=int)\n"
"parser.add_argument('in_height', type=int)\n"
"parser.add_argument('in_width', type=int)\n"
"parser.add_argument('num_filter', type=int)\n"
"parser.add_argument('ft_height', type=int)\n"
"parser.add_argument('ft_width', type=int)\n"
"parser.add_argument('st_height', type=int)\n"
"parser.add_argument('st_width', type=int)\n"
"parser.add_argument('padding', type=str, choices=['SAME', 'VALID'])\n"
"parser.add_argument('layout', type=str, choices=['NCHW', 'NHWC', 'HWNC'])\n"
"args = parser.parse_args()\n"
"batch = args.batch\n"
"in_channel = args.in_channel\n"
"in_height = args.in_height\n"
"in_width = args.in_width\n"
"num_filter = args.num_filter\n"
"ft_height = args.ft_height\n"
"ft_width = args.ft_width\n"
"st_height = args.st_height\n"
"st_width = args.st_width\n"
"padding = args.padding\n"
"layout = args.layout\n"
"if layout is 'HWCN'：\n"
"	A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')\n"
"elif layout is 'NHWC':\n"
"	A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')\n"
"elif layout is 'NCHW':\n"
"	A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')\n"
"else:\n"
"	raise ValueError('No definition for layout {}'.format(layout))\n"
"W = tvm.placeholder((ft_height, ft_width, in_channel, num_filter), name='W')\n"
"B = topi.nn.conv2d(A, W, [st_height, st_width], padding, layout)\n"
"if layout is 'HWCN':\n"
"	sch = tpoi.cuda.schedule_conv2d_hwcn(B)\n"
"elif layout is 'NCHW':\n"
"	sch = topi.cuda.schedule_conv2d_nchw(B)\n"
"else:\n"
"	raise ValueError('No schedule for layout {}'.format(layout))\n"
"ctx = tvm.context(device, 0)\n"
"a = tvm.nd.array(get_const_tuple(A.shape), ctx)\n"
"w = tvm.nd.array(get_const_tuple(W.shape), ctx)\n"
"with tvm.build_config(auto_unroll_max_step=1400, unroll_explicit=(device != 'cuda')):\n"
"	func = tvm.build(sch, [A, W, B], 'cuda')\n"
"	func(a, w, b)\n"
"	evaluator = func.time_evaluator(func.entry_name, ctx, number=100)\n"
"	print('Convolution: %f ms' % (evaluator(a, w, b).mean * 1e3))\n"



void test_tvm_conv2d(int a_n, int a_c, int a_h, int a_w,
				 	int w_c, int w_f, int w_h, int w_w,
				 	int o_h, int o_w)
{

}