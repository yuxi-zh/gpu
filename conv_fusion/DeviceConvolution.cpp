

class DevConvGen {
public:

	typedef function<size_t(size_t)> GetParamOffset;
	
	/* args : [
	 *	N, // batch
	 *	C, // in channel
	 * 	K, // out channel
	 *	D, H, W, // input shape
	 *	T, R, S, // filter shape 
	 *	pad_d, pad_h, pad_w, // padding
	 *	str_d, str_h, str_w, // stride
	 *	dil_d, dil_h, dil_w  // dilation
	 * ]
	 */
	DevConvGen(vector<unsigned> args);

	~DevConvGen();

	vector<unsigned> GetAuxiliaryArguments();

	dim3 GetGrid() { return grid; }

	dim3 GetBlock() { return block; }

	CUmodule GetCUDAModule(GetParamOffset offset);

private:

	string kernel_name;

	dim3 grid;

	dim3 block;

	vector<unsigned> kernel_args;
}

string AssembleKernelQueryCommond(vector<unsigned> args)
{
	ostringstream os;
	os << "python KernelQuery.py ";
	for (auto arg : args) os << arg << " "
	return os.str();
}

json QueryKernel(vector<unsigned> args)
{
	string command = AssemblePythonCommand();
	string query_result = RunCommand(command);
	return json::parse(query_result);
}

DevConvGen::DevConvGen(vector<unsigned> args)
{
	json query_result = QueryKernel(args);
	kernel_name = query_result["kernel_name"];
	grid = query_result["kernel_args"][0];
	block = query_result["kernel_args"][1];
	kernel_args = query_result["kernel_args"][2:];
}

string AssembleMaxasInsertionCommand(string name, string option)
{
	string sass_file = GetSassFile(name);
	string cubin_file = GetContainerCubinFile(name);
	ostringstream os;
	os << "maxas.pl -i -k";
	os << name << " " << option << " ";
	os << sass_file << " " << cubin_file;
	return os.str();
}

string AssembleConsMemRemapCommand(string name, vector<size_t> offset)
{
	string sass_file = GetSassFile(name);
	ostringstream os;
	os << "sed "
}

string AssembleBlockRemapCommand(string name)
{
	string sass_file = GetSassFile(name);
	ostringstream os;
	os <<
}

int DevConvGen::GetCUDAModule(GetParamOffset offset, CUmodule &module)
{
	vector<size_t> args_offset;
	for (int i = 0; i < kernel_args.size(); i++) 
		args_offset.push_back(offset(i));
	ostringstream os;
	os << AssembleConsMemRemapCommand(kernel_name, args_offset) << ";";
	os << AssembleBlockRemapCommand(kernel_name) << ";"
	os << AssembleMaxasInsertionCommand(kernel_name, option) << ";"
	RunCommand(os.str());
	LoadCUDAModule(GetDeviceCubinFile(name), module);
	return 0;
}