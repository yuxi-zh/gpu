#ifndef __DIRECT_CONV__
#define __DIRECT_CONV__

#include "conv_fusion_intf.h"

#include <vector>

class DirectConv : public NeonConvDevFunc {

	DirectConv(std::vector<unsigned> args);
	~DirectConv();

	int PrepareAuxiliaryParameter();

	int GenCUDAModule(bool vblock, 
		GetParamOffsetInConstMem offset, CUmodule &module);
}

#endif
