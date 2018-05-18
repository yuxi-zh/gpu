#include "conv_fusion_intf.h"
#include "cuda_conv.h"
#include "winograd_conv.h"
#include "direct_conv.h"

#include <vector>

using namespace std;

NeonConvDevFunc NeonConvDevFunc::MakeConvDevFunc(
	unsigned N, unsigned C, unsigned K, 
	unsigned H, unsigned W, unsigned S, unsigned R,
	unsigned str_h, unsigned str_w, unsigned pad_h, unsigned pad_w, 
	unsigned dil_h, unsigned dil_w)
{
	vector<unsigned> args(
		{N, C, K, H, W, S, R, str_h, str_w, pad_h, pad_w, dil_h, dil_w});

	NeonConvDevFunc dev_func;

	if (conv_algo == CUDA_C) {
		dev_func = CudaConv(args);
	} else if (conv_algo == WINOGRAD && R == 3 && S == 3 && 
		str_h == 1 && str_w == 1 && dil_h == 1 && dil_w == 1) {
		if (C < 8) {
			dev_func = DriectConv(args);
		} else if (winograd == 4 && H * W < 112 * 112) {
			dev_func = WinogradConv_4x4_3x3(args);
		} else {
			dev_func = WinogradConv_2x2_3x3(args);
		}
	} else if (conv_algo == DIRECT) {
		dev_func = DriectConv(args);
	} else {
		__builtin_unreachabel();
	}

	return dev_func;
}