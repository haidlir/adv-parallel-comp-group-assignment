// Author: Wesley Bland
// Copyright 2015 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.
//
// Example using MPI_Comm_split to divide a communicator into subcommunicators
//
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);

    // Get the rank and size in the original communicator
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    double mpi_time = 0.0;

    // Get the group of processes in MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Group incl_group, excl_group;
    MPI_Comm incl_comm, excl_comm;

    int* ranks = NULL;
    ranks = (int*) malloc(sizeof(int) * world_size/2);
    for (int i=0; i<world_size/2; i++)
      ranks[i] = i;

    mpi_time -= MPI_Wtime();
    MPI_Group_incl(world_group, world_size/2, ranks, &incl_group);
    MPI_Group_excl(world_group, world_size/2, ranks, &excl_group);

    // Create a new communicator based on the group
    MPI_Comm_create_group(MPI_COMM_WORLD, incl_group, 0, &incl_comm);
    MPI_Comm_create_group(MPI_COMM_WORLD, excl_group, 0, &excl_comm);
    mpi_time += MPI_Wtime();

    int incl_rank = -1, excl_rank = -1;
    int incl_size = -1, excl_size = -1;

    if (MPI_COMM_NULL != incl_comm) {
        MPI_Comm_rank(incl_comm, &incl_rank);
        MPI_Comm_size(incl_comm, &incl_size);
    }
    if (MPI_COMM_NULL != excl_comm) {
        MPI_Comm_rank(excl_comm, &excl_rank);
        MPI_Comm_size(excl_comm, &excl_size);
    }

    printf("INCL RANK/SIZE: %d/%d --- EXCL RANK/SIZE: %d/%d\n",
      incl_rank, incl_size, excl_rank, excl_size);

    MPI_Group_free(&world_group);
    MPI_Group_free(&incl_group);
    MPI_Group_free(&excl_group);

    if (MPI_COMM_NULL != incl_comm) {
      MPI_Comm_free(&incl_comm);
    }

    if (MPI_COMM_NULL != excl_comm) {
      MPI_Comm_free(&excl_comm);
    }

    free(ranks);
    if (world_rank == 0)
        printf("Counted time = %lf\n", mpi_time);
    MPI_Finalize();
}
