#include <stdio.h>
#include <stdlib.h>
// Include the MPI header 
#include <mpi.h> 
#include "win-gettimeofday.h"
void device_bubbleSort(int device_array[], int End) {//Used to sort the given array as a BubbleSort

	int swapped = 0;
	int temp;

	do {
		swapped = 0;
		for (int i = 0; i < End; i++) {
			if (&device_array[i] > &device_array[i + 1]) {
				temp = device_array[i];
				device_array[i] = device_array[i + 1];
				device_array[i + 1] = temp;
				swapped = 1;
			}
		}

	} while (swapped == 1);
}

void host_populateRandomArray(int x[], int num_elements) { //Used to populate the given array with random integers
	for (int i = 0; i < num_elements; i++) {
			x[i] = rand() % 100 + 1;
	}
}

int main(int argc, char* argv[])
{
	// Initialize the MPI environment 
	MPI_Init(NULL, NULL);

	// Get the number of processes 
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Get the rank of the process 
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	const int number_of_elements = 100; //Total amount of elements to be sorted
	int trials[number_of_elements]; //To be used if testing requires multiple attempts

	int size = trials[0] * sizeof(int); //Used to find the BYTE size, used to for memory allocation on the device and host

	int* host_array; //Used to store the array on the host
	
	double MPI_time_with_allocation; //Used to store time
	double MPI_end_time;//Used to store time
	



	if (world_size < 2) {
		fprintf(stderr, "World size must be greater than 1 for %s\n", argv[0]);
		int error = MPI_Abort(MPI_COMM_WORLD, 1);
		return error;
	}

	MPI_time_with_allocation = get_current_time();	
	if (world_rank == 0) {
		
		host_array = (int *)malloc(size);
		host_populateRandomArray(host_array, number_of_elements);
		MPI_Send(host_array, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		free(host_array);//Removes host_array memory assignment
	}
	else if (world_rank == 1) {
		int count;//Used to store the message size
		MPI_Status status;//Used to probe the incoming message
		MPI_Get_count(&status, MPI_INT, &count);//Used to get the message size before recieving it

		int* device_array = (int*)malloc(sizeof(int) * count); //Assign correct memory amount depending on incoming message size
		
		MPI_Recv(&device_array, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //Recieve the incomming message

		device_bubbleSort(device_array, number_of_elements);

		
	}
	MPI_Barrier(MPI_COMM_WORLD);//Blocks untill everything is completed

	MPI_end_time = get_current_time();//Marks the end time
	printf("Number of elements = %d, MPI Time (Including data transfer): %lfs\n", number_of_elements, (MPI_end_time - MPI_time_with_allocation));
	// Clean-up the MPI environment
	MPI_Finalize();

	return 0;
}