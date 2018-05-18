#include ""

class GlbFuncGenerator
{

}

/* demo to fuse convolution and gemm
 * FusedKernel(I, F, W, O) = Conv(I, F) * W
 */
int main(int argc, char const *argv[])
{
	GlbFuncGenerator gen;

	map<string, unsigned> conv_args = {
		{"N", 8}, {"C", 4}, {"K", 1}, {"H", 32}, {"W", 32}, 
		{"R", 3}, {"S", 3}, {"str_h", 1}, {"str_w", 1}, 
		{"pad_h", 0}, {"pad_w", 0}, {"dil_h", 1}, {"dil_w", 1}
	};
	auto conv = MakeNeonConvDevFunc(conv_args);
	gen.AddParameters(conv.param.begin(), conv.param.end());

	map<string, unsigned> gemm_args = {
		{"B", 8}, {"M", 30}, {"K", 30}, {"N", 16}
	};
	auto gemm = MakeNeonGemmDevFunc(gemm_args);
	gen.AddParameters(gemm.param.begin(), gemm.param.end());

	// merge params which denote ends of an edge in the dataflow graph
	gen.MergeParameter(conv.param["O"], gemm.param["I"]);
	// locate constant memory address for each parameter
	gen.ArrangeParameter();
	// call maxas assembler to build cubin files, then load into modules
	gen.BuildDevModules();
	// link global function module and device function modules
	Module module = gen.LinkModules();
	Function glbfunc = module.GetFunction();
	glbfunc.PrepareAuxiliaryParameters();
	glbfunc.Luanch({
		{"I", input}, {"F", filter}, {"W", weight}, {"O", output}
	});




	return 0;
}