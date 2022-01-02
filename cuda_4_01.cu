#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>

void worker_add(int *a, int *b, int *c, int tid, int NT, int N) {
	int size = (int) ceil(N/NT);
	for (int k = tid * size; k < (tid+1) * size; k++) {
		if (k >= N) {
			break;
		}
		c[k] = a[k] + b[k];
	}		
}

__global__ void add(int *a, int *b, int *c, int N) {
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] + b[tid];
}

int main( void ) {
	int NT = 4;
	int N = 10000;

	int* A = new int[N]();
	int* B = new int[N]();
	int* C = new int[N]();
	int* D = new int[N]();

	for (int i = 0; i < N; i++) {
		A[i] = i;
		B[i] = i % 3;
	}

	// Thread Version
	auto start = std::chrono::steady_clock::now();

	std::vector<std::thread> threads;

	for (int t = 0; t < NT; t++) {
		threads.push_back(std::thread(worker_add, A, B, C, t, NT, N));
	}

	for (auto& thread: threads) {
		thread.join();
	}

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << "Thread version takes " << diff.count() << " second" << std::endl;

	for (int i = 0; i < 5; i++) {
		std::cout << C[i] << " " << std::endl;
	}

	// GPU Version
	start = std::chrono::steady_clock::now();

	int *dev_A, *dev_B, *dev_D;
	cudaMalloc((void**) &dev_A, N * sizeof(int));
	cudaMalloc((void**) &dev_B, N * sizeof(int));
	cudaMalloc((void**) &dev_D, N * sizeof(int));

	cudaMemcpy(dev_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N, 1>>>(dev_A, dev_B, dev_D, N);

	cudaMemcpy(D, dev_D, N * sizeof(int), cudaMemcpyDeviceToHost);

	end = std::chrono::steady_clock::now();
	diff = end - start;
	std::cout << "GPU version takes " << diff.count() << " second" << std::endl;
	for (int i = 0; i < 5; i++) {
		std::cout << D[i] << " " << std::endl;
	}

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_D);
}
