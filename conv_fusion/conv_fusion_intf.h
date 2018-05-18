#ifndef __CONV_INTF__
#define __CONV_INTF__

class NeonConvDevFunc {
public:
	
	struct Parameter {

		// offset relative to the constant memory address of first parameter
		size_t offset;

		char value[8];
	}

	typedef std::function<size_t(Parameter)> GetParamOffsetInConstMem;

	virtual int GenCUDAModule(bool vblock, 
		GetParamOffsetInConstMem offset, CUmodule &module) = 0;

	static NeonConvDevFunc MakeConvDevFunc(
		unsigned N, unsigned C, unsigned K, 
		unsigned H, unsigned W, unsigned S, unsigned R,
		unsigned str_h, unsigned str_w, 
		unsigned pad_h, unsigned pad_w, 
		unsigned dil_h, unsigned dil_w);

protected:

	std::vector<Parameter> params;
}

#endif