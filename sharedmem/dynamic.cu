// __shared__ extern float shared_memory[];

__device__ void bar(float *in, float *out) {
	__shared__ float shared_memory[1024];
    shared_memory[threadIdx.x] = in[threadIdx.x] + 1;
    out[threadIdx.x] = shared_memory[threadIdx.x] + 1;
}

__global__ void foo(float *in, float *out) {
    // do something with shared_memory
	__shared__ float shared_memory[1024];
    shared_memory[threadIdx.x] = in[threadIdx.x];
    out[threadIdx.x] = shared_memory[threadIdx.x];
    bar(in, out);
}

int main() {
    float *in, *out;
    int shared_size = 1024; // can be decided in runtime
    foo<<<1, 1024, shared_size>>>(in, out);
}