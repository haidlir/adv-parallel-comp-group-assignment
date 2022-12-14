/**********************************************************************                                                                                      
 * MPI-based matrix multiplication AxB=C                                                                                                                     
 *********************************************************************/


#include "mpi.h"
#define N                 2048      /* number of rows and columns in matrix */
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
MPI_Status status;

double a[N][N],b[N][N],c[N][N];

main(int argc, char **argv)
{
  int numtasks,taskid,numworkers,source,dest,rows,offset,i,j,k;

  double timecounter_op, timecounter_all;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

  numworkers = numtasks-1;

  /*---------------------------- master ----------------------------*/
  if (taskid == 0) {
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 1.0;
        b[i][j]= 2.0;
      }
    }

    timecounter_all -= MPI_Wtime(); 
    timecounter_op -= MPI_Wtime();

    // Single Processor or serial
    if (numtasks == 1) {
        int i, j, k;
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                for (k = 0; k < N; k++) {
                    c[i][j] += a[i][k] * b[k][j]; //C = A*B; 
                }
            } // end middle loop
        } // end outer loop 
        timecounter_all += MPI_Wtime();
        fprintf(stdout,"Time MPI= %.6f\n", (timecounter_all));
        MPI_Finalize();
        return 0;
    }

    /* send matrix data to the worker tasks */
    rows = N/(numworkers+1);
    offset = rows;
    timecounter_op += MPI_Wtime();

    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b, N*N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      offset = offset + rows;
    }

    timecounter_op -= MPI_Wtime();
    // The master also helps the worker
    /* Matrix multiplication */
    for (k=0; k<N; k++) {
      for (i=0; i<rows; i++) {
        c[i][k] = 0.0;
        for (j=0; j<N; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
      }
    }
    
    timecounter_op += MPI_Wtime();
    /* wait for results from all worker tasks */
    for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }

    timecounter_all += MPI_Wtime();

    // printf("Here is the result matrix:\n");
    /*for (i=0; i<N; i++) {
      for (j=0; j<N; j++)
        printf("%6.2f   ", c[i][j]);
      printf ("\n");
    }*/

    fprintf(stdout,"Time Total= %.6f\n", (timecounter_all));
    double commtime = timecounter_all - timecounter_op; 
    fprintf(stdout,"Comm Time= %.6f\n",
           (commtime));
    fprintf(stdout,"Op Time= %.6f\n",
           (timecounter_op));

  }

  /*---------------------------- worker----------------------------*/
  if (taskid > 0) {
    source = 0;
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

    /* Matrix multiplication */
    for (k=0; k<N; k++)
      for (i=0; i<rows; i++) {
        c[i][k] = 0.0;
        for (j=0; j<N; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
      }

    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }


  MPI_Finalize();
}
