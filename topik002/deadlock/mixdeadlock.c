#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
        int rank,size;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);



        if(2!=size) MPI_Abort(MPI_COMM_WORLD, 1);

    int* buf1 = (int*)malloc(sizeof(int) * 10000);
    int* buf2 = (int*)malloc(sizeof(int) * 10000);

        buf1[0] = 1;
        buf2[0] = 1;
        if (0==rank) {
            MPI_Bcast(buf1,10000,MPI_INT,0,MPI_COMM_WORLD);
            MPI_Send(buf2,10000,MPI_INT,1,0, MPI_COMM_WORLD);
            printf("proc 0 done\n");
        }
        if (1==rank) {
            MPI_Recv(buf2,10000,MPI_INT,0,0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Bcast(buf1,10000,MPI_INT,0,MPI_COMM_WORLD);
            printf("proc 1 done\n");
        }
        MPI_Finalize();
}

/*The only change is increasing the number of communicated bytes by a factor of 10000. Why does this make the code deadlock?

Have a look at the communication modes part of the MPI documentation.
 A send can return successfully if a) there is a receive waiting on the receiving rank or b) 
 the receiving rank has a receive buffer of sufficient size allocated.

By default, MPI allocates a receive buffer of some size that will catch small receives such as the single ints you are sending out.*/