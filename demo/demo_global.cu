// demo_global.cu
__global__ void GlobalCpy(float *A, float *B, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float shared_memory[256];
	if (i < n) {
	    shared_memory[i] = A[i];
		B[i] = shared_memory[i];
	}
}