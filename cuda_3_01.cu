#include <iostream>

__global__ void kernel_empty( void ) {

}

__global__ void kernel_add(int a, int b, int *c) {
	*c = a + b;
}

int main( void ) {
	// Example 1: Invoke a kernel
	kernel_empty<<<1, 1>>>();
	printf("kernel executed\n");

	// Example 2: Simple addition
	int c;
	int *dev_c;

	cudaMalloc( (void**) &dev_c, sizeof(int) );
	kernel_add<<<1, 1>>>( 2, 7, dev_c );
	cudaMemcpy( &c, dev_c, sizeof(int), cudaMemcpyDeviceToHost );

	printf("2 + 7 = %d\n", c);
	cudaFree( dev_c );

	return 0;
}
