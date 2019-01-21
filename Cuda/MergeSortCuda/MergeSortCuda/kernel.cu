#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "win-gettimeofday.h"

/* Number of threads per block */
#define THREADS_PER_BLOCK 512

/*
* Kernel Function:  addWithThreadsAndBlocks
* --------------------
*  computes the element-wise sum of two 1D vectors of size num_elements by using the GPU
*  the kernel uses a one-dimensional grid of a one-dimensional block 
*  the one-dimensional block is composed of THREADS_PER_BLOCK number of threads
*
*  Input:    a -  int* (pointer to int) - first 1D vector of num_elements elements
*            b -  int* (pointer to int) - second 1D vector of num_elements elements
*            num_elements - int - number of elements composing the 1D vector
*
*  Output:   c - int* (pointer to int) - 1D vector resulting from the element-wise sum of a and b vectors
*/
__global__ void addWithThreadsAndBlocks(int *a, int *b, int num_elements, int *c)
{
	/* Compute the global unique index of each thread for a one-dimensional grid of a one-dimensional block of  THREADS_PER_BLOCK number of threads */
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	/* Perform the element-wise sum of the vectors 
	*  Note that the size of the input vectors is not multiple of the THREADS_PER_BLOCK number of threads.
	*  This means that some of the threads will not be involved in the computation.
	*  You have to esclude such threads
	*/
	if (index < num_elements) {

		c[index] = a[index] + b[index];

	}


}

__global__ void mergeSort(int *array) {
	int a = ((sizeof(*array)/2) + 1); //Gets half the size of the array.
	int b = (sizeof(*array) / 2);//Gets half the size of the array.

	int Left[a];  //Used as a temp array
	int Right[b]; //Used as a temp array

	for (int i = 0; i < a; i++) { //Copy half of the array into this array
		Left[i] = array[i];
	}
	for (int k = 0; k < b; k++) {
		Right[k] = array[(sizeof(*array) / 2) + 1 + k];  //Copy half of the array into this array
	}

	int j = 0;
	int h = 0;
	int k = 0;

	while (j < a && h < b) {
		if (Left[j] <= Right[h]) {
			array[k]
		}
	}
}

/*
* Function:  random_ints
* --------------------
*  generates a 1D vector of size equal to num_elements where each element is between 1 and 1000
*
*  Input:    num_elements - int - number of elements composing the 1D vector 
*
*  Output:   x - int* (pointer to int) - 1D vector of num_elements elements
*/
void random_ints(int* x, int num_elements)
{

	int i;

	for (i = 0; i < num_elements; i++)
	{
		x[i] = rand() % 1000 + 1;
	}
}

/*
* Function:  cpu_vector_sum
* --------------------
*  computes the element-wise sum of two 1D vectors of size num_elements by using ONLY the CPU
*
*  Input:    a -  int* (pointer to int) - first 1D vector of num_elements elements
*            b -  int* (pointer to int) - second 1D vector of num_elements elements
*            num_elements - int - number of elements composing the 1D vector
*
*  Output:   c - int* (pointer to int) - 1D vector resulting from the element-wise sum of a and b vectors
*/
void cpu_vector_sum(int *a, int *b, int num_elements, int *c)
{
	for (int i = 0; i < num_elements; i++)
	{
		c[i] = a[i] + b[i];
	}
}


/* In C, the "main" function is treated the same as every function, 
*  it has a return type (and in some cases accepts inputs via parameters). 
*  The only difference is that the main function is "called" by the operating 
*  system when the user runs the program. 
*  Thus the main function is always the first code executed when a program starts. 
*  This function returns an integer representing the application software status.
*/
int main(void)
{

	/* Number of trails */

	
	/* Number of elements of the 1D vectors for each trail */
	int trials[number_of_trials] = { 10, 100, 1000, 10000, 100000, 1000000, 10000000, 20000000, 30000000, 50000000, 60000000, 70000000, 80000000,  100000000 };

	/* Definition of the data structure used to memorize the time taken by the cpu to compute the element-wise sum of the vectors as the size of the vectors varies */
	double cpu_time_per_trials[number_of_trials];

	/* Definition of the data structure used to memorize the time taken by the gpu to compute the element-wise sum of the vectors 
	*  as the size of the vectors varies including the time to copy the vectors from/to HOST/DEVICE memory 
	*/
	double gpu_time_per_trials_with_data_transfer[number_of_trials];

	/* Definition of the data structure used to memorize the time taken by the gpu to compute the element-wise sum of the vectors
	*  as the size of the vectors varies not including the time to copy the vectors from/to HOST/DEVICE memory
	*/
	double gpu_time_per_trials_without_data_transfer[number_of_trials];

	/* Initialization an object of the type FILE, which will contain all the information necessary to control the stream */
	FILE *results;

	/* Definition of the COPY of the first 1D vector on the HOST as a pointer to integer */
	int* host_a;
	/* Definition of the COPY of the second 1D vector on the HOST as a pointer to integer */
	int* host_b;

	/* Definition of the COPY of the 1D vector on the HOST as a pointer to integer 
	*  whose elements are the sum of the elements of the first and the second vector 
	*/
	int* host_c;

	/* Definition of the 1D vector as pointer to integer whose elements are the sum of the elements of the first and the second vector computed by the CPU */
	int* CPU_host_c;

	/* Definition of the COPY of the first 1D vector on the DEVICE as a pointer to integer */
	int* device_a;

	/* Definition of the COPY of the second 1D vector on the DEVICE as a pointer to integer */
	int* device_b;

	/* Definition of the COPY of the 1D vector on the DEVICE as a pointer to integer whose elements are the sum of the elements of the first and the second vector */
	int* device_c;

	/* Definition of the start time of the computation of the element-wise sum of the vectors for the CPU */
	double cpu_start_time;

	/* Definition of the start time of the computation of the element-wise sum of the vectors for the CPU */
	double cpu_end_time;

	/* Definition of the start time of the computation of the element-wise sum of the vectors for the GPU
	*  which will also take into account data transfer time
	*/
	double gpu_start_time;

	/* Definition of the start time of the computation of the element-wise sum of the vectors for the GPU
	*  which will also take into account data transfer time
	*/
	double gpu_end_time;

	/* Definition of the start time of the computation of the element-wise sum of the vectors for the GPU
	*  which will NOT include data transfer time
	*/
	double gpu_start_time_no_data_transfer;

	/* Definition of the start time of the computation of the element-wise sum of the vectors for the GPU
	*  which will NOT include data transfer time
	*/
	double gpu_end_time_no_data_transfer;

	for (int k = 0; k < number_of_trials; k++)
	{
		/* Compute the amount of memory in bytes to be allocated to store the 1D vectors in both the HOST and the DEVICE memories */
		int size = trials[k] * sizeof(int);

		/* Dynamic allocation of the HOST memory to store the first 1D vector of size num_elements */
		host_a = (int *)malloc(size);

		/* Dynamic allocation of the HOST memory to store the first 1D vector of size num_elements */
		host_b = (int *)malloc(size);

		/* Dynamic allocation of the HOST memory to store the 1D vector of size num_elements 
		*  resulting from the element-wise sum of host_a and host_b 
		*/
		host_c = (int *)malloc(size);

		/* Dynamic allocation of the memory to store the 1D vector of size num_elements
		*  resulting from the element-wise sum of host_a and host_b performed by the CPU
		*/
		CPU_host_c = (int *)malloc(size);

		/* Dynamic allocation of the HOST memory to store the first 1D vector of size num_elements */
		cudaMalloc((void **)&device_a, size);

		/* Dynamic allocation of the HOST memory to store the first 1D vector of size num_elements */
		cudaMalloc((void **)&device_b, size);

		/* Dynamic allocation of the HOST memory to store the 1D vector of size num_elements
		*  resulting from the element-wise sum of host_a and host_b
		*/
		cudaMalloc((void **)&device_c, size);

		/* Fill the elements of the first 1d vector stored in the HOST memory using random_ints function */
		random_ints(host_a, trials[k]);

		/* Fill the elements of the second 1d vector stored in the HOST memory using random_ints function */
		random_ints(host_b, trials[k]);

		/* Invocation of the function in the external library responsible of taking the starting time of the compuation for the CPU */
		cpu_start_time = get_current_time();

		/* Invocation of the function  cpu_vector_sum which computes the element-wise sum of the input vectors on the CPU */
		cpu_vector_sum(host_a, host_b, trials[k], CPU_host_c);

		/* Invocation of the function in the external library responsible of taking the starting time of the compuation for the CPU */
		cpu_end_time = get_current_time();

		/* Print on the Standard Output the performance in term of time of the CPU in computing 
		*  the element-wise sum as the number of elements composing the vectors increases 
		*/
		printf("Trail: %d - Number of vector elements: %d - CPU Time: %lfs\n", k, trials[k], cpu_end_time - cpu_start_time);

		cpu_time_per_trials[k] = cpu_end_time - cpu_start_time;

		/* Invocation of the function in the external library responsible of taking the starting time of the compuation for the GPU 
		*  including time needed to copy the 1D vectors from HOST memory to DEVICE memory and from DEVICE memory to HOST memory 
		*/
		gpu_start_time = get_current_time();

		/* Copy the first 1D vector from HOST memory to DEVICE memory */
		cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

		/* Copy the second 1D vector from HOST memory to DEVICE memory */
		cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

		/* Definition of the number of threads of the 1D block */
		dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
		
		/* Definition of the number of blocks of the 1D grid */
		dim3 dimGrid((trials[k] + dimBlock.x - 1) / dimBlock.x, 1, 1);

		/* Invocation of the function in the external library responsible of taking the starting time of the compuation for the GPU
		*  NOT including time needed to copy the 1D vectors from HOST memory to DEVICE memory and from DEVICE memory to HOST memory
		*/
		gpu_start_time_no_data_transfer = get_current_time();

		printf("GPU Start Time (NOT Including Data Transfer): %lfs\n", gpu_start_time_no_data_transfer);

		/* Launch addWithThreadsAndBlocks kernel on GPU with dimGrid grids and dimBlock blocks */
		addWithThreadsAndBlocks << < dimGrid, dimBlock >> >(device_a, device_b, trials[k], device_c);

		/*  Handling function of the CUDA runtime application programming interface.
		*   Returns the last error from a runtime call.
		*/
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			/* Returns the description string for an error code */
			printf("Error: %s\n", cudaGetErrorString(error));
		}

		/* Invocation of the function in the external library responsible of taking the starting time of the compuation for the GPU
		*  NOT including time needed to copy the 1D vectors from HOST memory to DEVICE memory and from DEVICE memory to HOST memory
		*/
		gpu_end_time_no_data_transfer = get_current_time();

		printf("GPU End Time (NOT Including Data Transfer): %lfs\n", gpu_end_time_no_data_transfer);

		/* Copy back the result of the element-wise sum of the first and the second 1D vectors computed by the DEVICE from DEVICE memory to HOST memory */
		cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

		/* Invocation of the function in the external library responsible of taking the starting time of the compuation for the GPU 
		*  including time needed to copy the 1D vectors from HOST memory to DEVICE memory and from DEVICE memory to HOST memory 
		*/
		gpu_end_time = get_current_time();

		/* Print on the Standard Output the performance in term of time of the GPU in computing
		*  the element-wise sum as the number of elements composing the vectors increases
		*/
		printf("Trail: %d - Number of vector elements: %d - GPU Time (NOT Including Data Transfer): %lfs\n", k, trials[k], gpu_end_time_no_data_transfer - gpu_start_time_no_data_transfer);
		printf("Trail: %d - Number of vector elements: %d - GPU Time (Including Data Transfer): %lfs\n", k, trials[k], gpu_end_time - gpu_start_time);

		gpu_time_per_trials_without_data_transfer[k] = gpu_end_time_no_data_transfer - gpu_start_time_no_data_transfer;
		gpu_time_per_trials_with_data_transfer[k] = gpu_end_time - gpu_start_time;

		/* Deallocation of the HOST memory previously allocated by malloc storing the first 1D vector */
		free(host_a);

		/* Deallocation of the HOST memory previously allocated by malloc storing the second 1D vector */
		free(host_b);

		/* Deallocation of the HOST memory previously allocated by malloc storing the result of the element-wise sum */
		free(host_c);

		/* Deallocation of the memory previously allocated by malloc storing the the result of the element-wise sum performed by the CPU */
		free(CPU_host_c);

		/* Deallocation of the DEVICE memory previously allocated by cudaMalloc storing the first 1D vector */
		cudaFree(device_a);

		/* Deallocation of the DEVICE memory previously allocated by cudaMalloc storing the second 1D vector */
		cudaFree(device_b);

		/* Deallocation of the DEVICE memory previously allocated by cudaMalloc storing the result of the element-wise sum */
		cudaFree(device_c);

		/* 	Destroy all allocations and reset all state on the current device in the current process. */
		cudaDeviceReset();
	}

	/* Performance in terms of computational time are stored in a CSV file */
	results = fopen("results.csv", "w");
	if (results != NULL)
	{
		fprintf(results,"Trial,Vector Size, CPU Time, GPU Time No Data Transfer, GPU Time Data Transfer\n");
		for (int k = 0; k < number_of_trials; k++)
		{
			fprintf(results, "%d,%d,%lf,%lf,%lf\n", k, trials[k], cpu_time_per_trials[k], gpu_time_per_trials_without_data_transfer[k], gpu_time_per_trials_with_data_transfer[k]);
		}
		fclose(results);
	}
	else
	{
		printf("ERROR: FILE NOT OPENED");
		return -1;
	}

	return 0;
}
