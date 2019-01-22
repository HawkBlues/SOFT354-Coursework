#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "win-gettimeofday.h"

/* Number of threads per block */
#define THREADS_PER_BLOCK 512





__global__ void bubbleSort(int array[],  int Middle, int End) {
	int swapped = 0;
	int temp;
	do {
		swapped = 0;
		for (int i = 0; i < End; i++) {
			if (array[i] > array[i + 1]) {
				temp = array[i];
				array[i] = array[i + 1];
				array[i + 1] = temp;
				swapped == 1;
			}
		}
	
	} while (swapped == 1);
			
}




void populateRandomArray(int *x, int num_elements) {
	for (int i = 0; i < num_elements; i++) {
		x[i] = rand() % 100 + 1;
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
	const int number_of_trials = 100000;


	int trials[number_of_trials];

	int* host_a; //used to store the whole 1d matrix
	int* host_c;//TESTING

	int* device_a;
	int* device_c;//TESTING

	double cpu_start_time;
	double cpu_time_without_transfer;
	double cpu_time_with_transfer;

	
	double gpu_time_without_transfer;
	double gpu_time_with_transfer;
	double gpu_end_time_without_transfer;
	double gpu_end_time_with_transfer;



	//for (int i = 0; i < number_of_trials; i++) {
	for (int i = 0; i < 1; i++) {
		int size = trials[i] * sizeof(int);
	

		int middle = (number_of_trials / 2); //Used to find the middle of the matrix
		int end = number_of_trials;//Used to find the end of the matrix

		host_a = (int *)malloc(size);//Used to store the while 1d matrix
		host_c = (int *)malloc(size);//TESTING
		
		cudaMalloc((void **)&device_a, size);
		cudaMalloc((void **)&device_c, size); //TESTING

		populateRandomArray(host_a, number_of_trials);

		
		gpu_time_with_transfer = get_current_time(); //Gets time before the Memory allocation

		cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);

		gpu_time_without_transfer = get_current_time(); //Gets time after the memory allocation

		dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
		dim3 dimGrid((trials[i] + dimBlock.x - 1) / dimBlock.x, 1, 1);
	
		bubbleSort << < dimGrid, dimBlock >> > (device_a, middle, end);
	

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			/* Returns the description string for an error code */
			printf("Error: %s\n", cudaGetErrorString(error));
		}

		cudaThreadSynchronize();

		gpu_end_time_without_transfer = get_current_time();

		cudaMemcpy(host_c, device_a, size, cudaMemcpyDeviceToHost);

		gpu_end_time_with_transfer = get_current_time();

		printf("Number of elements = %d, GPU Time (Not including data transfer): %lfs\n", number_of_trials, (gpu_end_time_without_transfer - gpu_time_without_transfer));
		printf("Number of elements = %d, GPU Time (Including data transfer): %lfs\n", number_of_trials, (gpu_end_time_with_transfer - gpu_time_with_transfer));



	
		free(host_a);
		free(host_c); //TESTING

		cudaFree(device_a);
		cudaFree(device_c);//TESTING

		cudaDeviceReset();
	}
	return 0;
}
