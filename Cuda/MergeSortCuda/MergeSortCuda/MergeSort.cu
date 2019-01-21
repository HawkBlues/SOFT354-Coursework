#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "win-gettimeofday.h"

/* Number of threads per block */
#define THREADS_PER_BLOCK 512


__global__ void mergeSortLeft(int *array, int *Left, int *Right, int Middle, int End) {
	for (int i = 0; i <= Middle; i++) { //Copy half of the array into the Left
		Left[i] = array[i];
	}
}

__global__ void mergeSortRight(int *array, int *Left, int *Right, int Middle, int End) {
	for (int i = (Middle + 1); i <= End; i++) {
		Right[i] = array[i];  //Copy the second half of the array into the Right
	}
}

__global__ void mergeBoth(int *array, int *Left, int *Right, int Middle, int End) {
	int i = 0;
	int j = 0;
	int k = 0;

	while (i < Middle && j < End) {
	if (Left[i] <= Right[j]) {
		array[k] = Right[j];
		j++;
	}
	else {
		array[k] = Left[i];
		i++;
	}
	k++;
}
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
	const int number_of_trials = 20;


	int trials[number_of_trials];

	int* host_a; //used to store the whole 1d matrix
	int* host_Left; //Used to store half of the matrix
	int* host_Right; //Used to store the other half of the matrix

	int* host_c;//TESTING

	int* device_a;
	int* device_Left;
	int* device_Right;

	int* device_c;//TESTING


	//for (int i = 0; i < number_of_trials; i++) {
	for (int i = 0; i < 1; i++) {
		int size = trials[i] * sizeof(int);

		int middle = (number_of_trials / 2); //Used to find the middle of the matrix
		int end = number_of_trials;//Used to find the end of the matrix

		host_a = (int *)malloc(size);//Used to store the while 1d matrix
		host_Left = (int *)malloc(size);//Used to get the si
		host_Right = (int *)malloc(size);//Used to store the right half of the matrix

		host_c = (int *)malloc(size);//TESTING

		cudaMalloc((void **)&device_a, size);
		cudaMalloc((void **)&device_Left, (size));
		cudaMalloc((void **)&device_Right, (size));
		cudaMalloc((void **)&device_c, (size)); //TESTING

		populateRandomArray(host_a, number_of_trials);



		cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(device_Left, host_Left, size, cudaMemcpyHostToDevice);
		cudaMemcpy(device_Right, host_Right, size, cudaMemcpyHostToDevice);

		dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);
		dim3 dimGrid((trials[i] + dimBlock.x - 1) / dimBlock.x, 1, 1);

		printf("Entire Sorted List");
		for (int i = 0; i < number_of_trials; i++) {
			printf("%d,", host_a[i]);
		}
		printf("\n");
	

		mergeSortLeft << < dimGrid, dimBlock >> > (device_a, device_Left, device_Right, middle, end);
		cudaThreadSynchronize();
		mergeSortRight << < dimGrid, dimBlock >> > (device_a, device_Left, device_Right, middle, end);
		cudaThreadSynchronize();
		//mergeBoth << < dimGrid, dimBlock >> > (device_c, device_Left, device_Right, middle, end);
		//When LEft and right are run, they populate device_left and right correctly. This is copied correctly.
		//Yet when trying to copy DeviceC (used for testing) from mergeBoth to see the contents, it's totally blank?

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			/* Returns the description string for an error code */
			printf("Error: %s\n", cudaGetErrorString(error));
		}

		cudaThreadSynchronize();

		cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);
	

		printf("Entire Sorted List");
		for (int i = 0; i < number_of_trials; i++) {
			printf("%d,", host_c[i]);
		}
		printf("\n");


	
		free(host_a);
		free(host_Right);
		free(host_Left);
		free(host_c); //TESTING

		cudaFree(device_a);
		cudaFree(device_Left);
		cudaFree(device_Right);
		cudaFree(device_c);//TESTING

		cudaDeviceReset();
	}
	return 0;
}
