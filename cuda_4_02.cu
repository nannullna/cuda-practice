#include <iostream>
#include <chrono>

#define N 4096
#define BLOCK 32

__global__ void MatMul(float* MatA, float* MatB, float* MatC)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i < N) && (j < N)) {
		float _c = 0.0f;
		for (int k = 0; k < N; k++) {
			_c += MatA[i*N+k] * MatB[k*N+j];
		}
		MatC[i*N+j] = _c;
	}
}

__global__ void MatMulShared(float* MatA, float* MatB, float* MatC)
{
	const int bx = BLOCK;
	const int by = BLOCK;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int i = blockIdx.x * bx + tx;
	const int j = blockIdx.y * by + ty;
	const int gx = gridDim.x;
	const int gy = gridDim.y;

//	printf("bx:%d, by:%d, tx:%d, ty:%d, i:%d, j:%d, gx:%d, gy:%d\n", bx, by, tx, ty, i, j, gx, gy);

	__shared__ float shared_A[BLOCK][BLOCK];
	__shared__ float shared_B[BLOCK][BLOCK];

	float _c = 0.0f;

	for (int k = 0; k < gy; k++) {

		if ((i < N) && (j < N)) {
			shared_A[ty][tx] = MatA[(i) * N + (k*by + ty)];
			shared_B[tx][ty] = MatB[(k*bx + tx) * N + (j)];
		}
		__syncthreads();

		for (int kk = 0; kk < bx; kk++) {
			_c += shared_A[kk][tx] * shared_B[kk][ty];
		}
		__syncthreads();

		MatC[i*N+j] = _c;
	}
}

__global__ void MatMulShared2(float* MatA, float* MatB, float* MatC)
{
	const int bx = BLOCK;
	const int by = BLOCK;
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int i = blockIdx.x * bx + tx;
	const int j = blockIdx.y * by + ty;
	const int gx = gridDim.x;
	const int gy = gridDim.y;

	__shared__ float shared_A[BLOCK][BLOCK];
	__shared__ float shared_B[BLOCK][BLOCK];

	if ((i < N) && (j < N)) {
		float _c = 0.0f;

		for (int k = 0; k < gy; k++) {
			shared_A[ty][tx] = MatA[i * N + (k*by+ty)];
			shared_B[tx][ty] = MatB[(k*bx+tx) * N + j];
			__syncthreads();

			for (int kk = 0; kk < bx; kk++) {
				_c += shared_A[kk][tx] * shared_B[kk][ty];
			}
			__syncthreads();
		}
		MatC[i*N+j] = _c;
	}
}

__global__ void MatMulKernel(float* MatA, float* MatB, float* MatC, int Width) {
	__shared__ float TileA[BLOCK][BLOCK];
	__shared__ float TileB[BLOCK][BLOCK];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * BLOCK + ty;
	int Col = bx * BLOCK + tx;
	float Pvalue = 0.0f;

	for (int m = 0; m < Width/BLOCK; ++m) {
		TileA[ty][tx] = MatA[Row*Width + m * BLOCK + tx];
		TileB[ty][tx] = MatB[(m * BLOCK + ty) * Width + Col];
		__syncthreads();

		for (int k = 0; k < BLOCK; ++k) {
			Pvalue += TileA[ty][k] * TileB[k][tx];
		}
		__syncthreads();
	}
	MatC[Row*Width + Col] = Pvalue;
}


int main(int argc, char** argv)
{
	float* host_A = new float[N*N];
	float* host_B = new float[N*N];
	float* host_C = new float[N*N];
	float* host_D = new float[N*N];

	float* dev_A;
	float* dev_B;
	float* dev_D;

	// initialization
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			host_A[i*N+j] = (i + j) % 5;
			host_B[i*N+j] = (i - j) % 7;
		}
	}

	// print matrix A, B
//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < N; j++) {
//			std::cout << host_A[i*N+j] << " ";
//		}
//		std::cout << std::endl;
//	}

//	for (int i = 0; i < N; i++) {
//		for (int j = 0; j < N; j++) {
//			std::cout << host_B[i*N+j] << " ";
//		}
//		std::cout << std::endl;
//	}

	dim3 blockDim(BLOCK, BLOCK);
	int gx = (N % blockDim.x == 0) ? N/blockDim.x : N/blockDim.x + 1;
	int gy = (N % blockDim.y == 0) ? N/blockDim.y : N/blockDim.y + 1;
	dim3 gridDim(gx, gy);

	auto start = std::chrono::steady_clock::now();

	if (atoi(argv[1]) == 1)

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < N; k++) {
				host_C[i*N+j] += host_A[i*N+k] * host_B[k*N+j];
			}
		}
	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "CPU time: " << diff.count() << std::endl;

	start = std::chrono::steady_clock::now();

	cudaMalloc((void**) &dev_A, N*N*sizeof(float));
	cudaMalloc((void**) &dev_B, N*N*sizeof(float));
	cudaMalloc((void**) &dev_D, N*N*sizeof(float));

	end = std::chrono::steady_clock::now();
	diff = end - start;
	std::cout << "cudaMalloc time: " << diff.count() << std::endl;

	start = std::chrono::steady_clock::now();

	cudaMemcpy(dev_A, host_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, host_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

	end = std::chrono::steady_clock::now();
	diff = end - start;
	std::cout << "cudaMemcpy time: " << diff.count() << std::endl;

	start = std::chrono::steady_clock::now();

	if (atoi(argv[2]) == 0)
	MatMul<<<gridDim, blockDim>>>(dev_A, dev_B, dev_D);

	if (atoi(argv[2]) == 1)
	MatMulShared<<<gridDim, blockDim>>>(dev_A, dev_B, dev_D);
	
	if (atoi(argv[2]) == 2)
	MatMulShared2<<<gridDim, blockDim>>>(dev_A, dev_B, dev_D);

	if (atoi(argv[2]) == 3)
	MatMulKernel<<<gridDim, blockDim>>>(dev_A, dev_B, dev_D, N);

	end = std::chrono::steady_clock::now();
	diff = end - start;
	std::cout << "GPU kernel time: " << diff.count() << std::endl;

	start = std::chrono::steady_clock::now();

	cudaMemcpy(host_D, dev_D, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	end = std::chrono::steady_clock::now();
	diff = end - start;
	std::cout << "cudaMemcpy time: " << diff.count() << std::endl;

	int err_count = 0;
	float err;
	float epsilon = 0.001;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			err = abs(host_C[i*N+j] - host_D[i*N+j]);
			if (err > epsilon) {
				if (err_count < 5) {
					std::cout << "C" << i << ", " << j << ": " << host_C[i*N+j] << std::endl;
					std::cout << "D" << i << ", " << j << ": " << host_D[i*N+j] << std::endl;
				}
				err_count++;
			}
		}
	}
	std::cout << "Error: " << err_count << std::endl;

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_D);
}
