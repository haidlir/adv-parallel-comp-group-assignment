/*
Forked from https://hpc.nmsu.edu/discovery/mpi/programming-with-mpi/
Accessed on October 7th 2022 by Haidlir Naqvi
*/

#include <mpi.h>
#include "stdio.h"

int main(int argc, char** argv)
{
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);

	// Get the rank of the process
	int PID;
	MPI_Comm_rank(MPI_COMM_WORLD, &PID);

	// Get the number of processes
	int number_of_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);

	// Get the name of the processor
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_length;
	MPI_Get_processor_name(processor_name, &name_length);

	// Print off a hello world message
	printf("Hello MPI user: from process PID %d out of %d processes on machine %s\n", PID, number_of_processes, processor_name);

	// Finalize the MPI environment
	MPI_Finalize();

	return 0;
}