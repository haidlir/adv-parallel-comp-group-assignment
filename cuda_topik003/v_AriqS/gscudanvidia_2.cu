#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX_ITER 1000000
#define MAX 100 //maximum value of the matrix element
#define TOL 0.000001


void print_matrix(float ***N, int rows, int columns,int n) {

  int i, j;
  //print only 4*4 matrix.
  if (rows > 4)
    rows = 4;
  if (columns > 4)
    columns = 4;
  printf("Printing only first 4 results \n");
  for (i = 0; i < rows; i++) {
    printf("\n");
    for (j = 0; j < columns; j++) {
    
        printf("%lf \t", (*N)[i][j]); //row major accessing with red color.
    

    }
    
  }
  
}



// Generate a random float number with the maximum value of max
float rand_float(int max){
  return ((float)rand()/(float)(RAND_MAX)) * max;
}

// Allocate 2D matrix
void allocate_init_2Dmatrix(float ***mat,  int n, int m){
  int i, j;
  *mat = (float **) malloc(n * sizeof(float *));
  for(i = 0; i < n; i++) {
    (*mat)[i] = (float *)malloc(m * sizeof(float));
    for (j = 0; j < m; j++)
      (*mat)[i][j] = rand_float(MAX);
  }

}

// solver
  __global__ void solver(float ***mat, int n, int m){
  float diff = 0, temp;
  int done = 0, cnt_iter = 0, i, j;

  while (!done && (cnt_iter < MAX_ITER)){
    diff = 0;
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++){
	temp = (*mat)[i][j];
	(*mat)[i][j] = 0.2 * ((*mat)[i][j] + (*mat)[i][j - 1] + (*mat)[i - 1][j] + (*mat)[i][j + 1] + (*mat)[i + 1][j]);
	diff += abs((*mat)[i][j] - temp);
      }
    if (diff/n/n < TOL)
      done = 1; 
    printf("diff : %f \n",diff);
    cnt_iter ++;
  }
  


  if (done){
    printf("Solver converged after %d iterations\n", cnt_iter);
    }
  else{printf("Solver not converged after %d iterations\n", cnt_iter);}
    
}

int main(int argc, char *argv[]) {
  int n;
  float **a,**ad;

  if (argc < 2) {
    printf("Call this program with two parameters: matrix_size communication \n");
    printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");

    exit(1);
  }

  n = atoi(argv[1]);
  int blocksize = atoi(argv[2]);
  float *temph[n];
  printf("Matrix size = %d\n", n);
  allocate_init_2Dmatrix(&a, n, n);

float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
// Allocate 2D array in Device
  cudaMalloc((void **)&ad,n*sizeof(float *));
 for (int i = 0; i < n; i++){
        cudaMalloc(&temph[i], n*sizeof(float));
}
  cudaMemcpy(ad,temph,n*sizeof(float *),cudaMemcpyHostToDevice);

 for (int i = 0; i < n; i++){
  cudaMemcpy(temph[i],a[i],n*sizeof(float),cudaMemcpyHostToDevice);
}



  dim3 DimBlock(blocksize,blocksize);
  dim3 DimGrid(1,1);
solver<<<1, 1>>>(&ad, n, n,true);
cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
cudaMemcpy(a,&ad,n*n*sizeof(float),cudaMemcpyDeviceToHost);
cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) 
    printf("Error1: %s\n", cudaGetErrorString(err1));
cudaFree(ad);

printf(">> Num of Block = 1 | Block Dim = 1 |Matrix size = %d\n", n);
 cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
     print_matrix(&a,n,n,n); 
    printf("Time elapsed on GPU: %f ms.\n\n",  gpu_elapsed_time_ms);
    
  return 0;
}