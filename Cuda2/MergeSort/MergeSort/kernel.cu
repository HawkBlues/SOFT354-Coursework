#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#include "win-gettimeofday.h"

#define THREADS_PER_BLOCK 512

__global__ void mergeSort(int *array, int *Left, int *Right, int Middle, int End) {
	//int a = (*Middle + 1); //Gets half the size of the array.
	//int b = (sizeof(*array) / 2);//Gets half the size of the array.


	for (int i = 0; i <= Middle; i++) { //Copy half of the array into the Left
		Left[i] = array[i];
	}
	for (int k = (Middle + 1); k <= End; k++) {
		Right[k] = array[Middle + 1 + k];  //Copy the second half of the array into the Right
	}

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

	//for (int i = 0; i < End; i++) {
	//	printf("%d,", array[i]);
	//}

}

void populateRandomArray(int *x, int num_elements) {
	for (int i = 0; i <= num_elements; i++) {
		x[i] = rand() % 100 + 1;
	}
}

int main(void) {

	const int number_of_trials = 5;

	int trials[number_of_trials] = { 10, 100, 1000, 10000, 100000,};

	int* host_a; //used to store the whole 1d matrix
	int* host_Left; //Used to store half of the matrix
	int* host_Right; //Used to store the other half of the matrix
	int* host_Middle;//Stores the middle location of the matrix
	int* host_End;//Stores the end location of the matrix

	int* device_a;
	int* device_Left;
	int* device_Right;
	int* device_Middle;
	int* device_End;

	for (int i = 0; i < number_of_trials; i++) {

		int size = trials[i] * sizeof(int);
		int middle = (number_of_trials / 2); //Used to find the middle of the matrix
		int end = number_of_trials;//Used to find the end of the matrix

		host_a = (int *)malloc(size);//Used to store the while 1d matrix
		host_Left = (int *)malloc(size);//Used to get the si
		host_Right = (int *)malloc(size);//Used to store the right half of the matrix

		cudaMalloc((void **)&device_a, size);
		cudaMalloc((void **)&device_Left, size);
		cudaMalloc((void **)&device_Right, size);

		//populateRandomArray(host_a, trials[i]);

		free(host_a);
		free(host_Right);
		free(host_Left);

		cudaFree(device_a);
		cudaFree(device_Left);
		cudaFree(device_Right);
		cudaDeviceReset();

	}





	return 0;
}