/**********************************************************************                                                                                      
 * MPI-based matrix multiplication AxB=C                                                                                                                     
 *********************************************************************/


#include "mpi.h"
//#define N                 256      /* number of rows and columns in matrix */
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
MPI_Status status;



int main(int argc, char *argv[1])
{
  int numtasks,taskid,numworkers,source,dest,rows,offset,i,j,k,N;

  

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  N = atoi(argv[1]);
  double a[N][N],b[N][N],c[N][N];
  numworkers = numtasks-1;
 

for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        a[i][j]= 1.0;
        b[i][j]= 2.0;
      }
    }
   double tscom1 = MPI_Wtime();
  /*---------------------------- master ----------------------------*/
  if (taskid == 0) {
    


    /* send matrix data to the worker tasks */
   
    rows = N/numworkers;
    offset = 0;

    for (dest=1; dest<=numworkers; dest++)
    {
      MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
      MPI_Send(&a[offset][0], rows*N, MPI_DOUBLE,dest,1, MPI_COMM_WORLD);
      MPI_Send(&b, N*N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
      offset = offset + rows;
    }

   
    /* wait for results from all worker tasks */
    

   

    

  } else{
    source = 0;
    MPI_Recv(&offset, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);
    MPI_Recv(&b, N*N, MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &status);

    /* Matrix multiplication */
    
  }

  double tfcom1 = MPI_Wtime();
	double tsop = MPI_Wtime();
  printf("start op time: %f\n", tsop);
  /*---------------------------- worker----------------------------*/
  if (taskid > 0) {
   for (k=0; k<N; k++)
      for (i=0; i<rows; i++) {
        c[i][k] = 0.0;
        for (j=0; j<N; j++)
          c[i][k] = c[i][k] + a[i][j] * b[j][k];
      }
  }
  double tfop = MPI_Wtime();
  printf("end op time: %f\n", tfop);
	double tscom2 = MPI_Wtime();


  /// recv func
   if (taskid == 0) {
  for (i=1; i<=numworkers; i++)
    {
      source = i;
      MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, &status);
      MPI_Recv(&c[offset][0], rows*N, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, &status);
    }
   }else{
    MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    MPI_Send(&c, rows*N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
   }
  double tfcom2 = MPI_Wtime();

  if (taskid == 0) {
		float com_time = (tfcom1-tscom1) + (tfcom2-tscom2);
		float ops_time = tfop - tsop;
		float total_time = com_time + ops_time;

		printf("Communication time: %f\n", com_time);
		printf("Operations time: %f\n", ops_time);
		printf("Total time: %f\n", total_time);

		FILE *f;
		if (access("results.csv", F_OK) == -1) {
 			f = fopen("results.csv", "a");
			fprintf(f, "Communication-time;Operations-time;Total-time;\n");
		}
		else {
			f = fopen("results.csv", "a");
		}

		fprintf(f, "%f;%f;%f;\n",  com_time, ops_time, total_time);
		fclose(f);
	}

  MPI_Finalize();
  return 0;
}
