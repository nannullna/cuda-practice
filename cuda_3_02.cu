#include <iostream>

void kernel_info( void ) {

}

int main( void ) {
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount( &count );
	// get the device count

	for (int i = 0; i < count; i++) {
		cudaGetDeviceProperties( &prop, i );
		printf("====== Information for Device %d ======\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);

		printf("====== Memory Information for Device %d ======\n", i);
		printf("Total global memory: %ld\n", prop.totalGlobalMem);
		printf("Total constant memory: %ld\n", prop.totalConstMem);
		printf("Shared Memory per Block: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per Block (32bit): %d\n", prop.regsPerBlock);
		printf("Warp size (Threads in Warp): %d\n", prop.warpSize);
		printf("Max threads per Block: %d\n", prop.maxThreadsPerBlock);

		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	}

	// Fill in the desired level
	cudaDeviceProp desired_prop;
	memset(&desired_prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	// assume that we need at least version 1.3

	int dev;
	cudaGetDevice(&dev);
	printf("ID of current CUDA device: %d\n", dev);

	cudaChooseDevice(&dev, &prop);
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
	cudaSetDevice(dev);
	printf("CUDA device set\n");
}
