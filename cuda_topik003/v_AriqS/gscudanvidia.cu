#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX 100
#define MAX_ITER 13
#define TOL 0.000001

// Generate a random float number with the maximum value of max

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


__global__ void solver(float **matdi, float **matdo, int n, int m, float *diff){

 int i= blockIdx.y*blockDim.y + threadIdx.y;
 int j= blockIdx.x*blockDim.x + threadIdx.x;

  if( (i > 0) && (j > 0) && (i < (n-1)) && (j <(m-1))){
        (matdo)[i][j] = 0.2 * ((matdi)[i][j] + (matdi)[i][j-1 ] + (matdi)[i-1 ][j] + (matdi)[i][j + 1] + (matdi)[i + 1][j]);
        atomicAdd(diff, abs((matdo)[i][j] - (matdi)[i][j]));
      }


}


int main(int argc, char *argv[]) {

  int n, cnt_iter=0;
  float **a,**adi, **ado, *d_diff;
  if (argc < 2) {
    printf("Call this program with two parameters: matrix_size communication \n");
    printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
    exit(1);
    }

  n = atoi(argv[1]);
  int blocksize = atoi(argv[2]);
 // float **temp;
  float **temi, **temo;
  temi = new float*[n];
  temo = new float*[n];
  printf("Matrix size = %d\n", n);
  allocate_init_2Dmatrix(&a, n, n);
  print_matrix(&a,n,n,n); 
// Allocate 2D array in Device
     float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);

  cudaMalloc(&adi, n*sizeof(float *));
  cudaMalloc(&ado, n*sizeof(float *));
  for(int i=0;i<n;i++){
    cudaMalloc(&(temi[i]),n*sizeof(float));
    cudaMalloc(&(temo[i]),n*sizeof(float));
    cudaMemcpy(temi[i],a[i],n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(temo[i],a[i],n*sizeof(float),cudaMemcpyHostToDevice);
    }
  cudaMemcpy(adi,temi,n*sizeof(float *),cudaMemcpyHostToDevice);
  cudaMemcpy(ado,temo,n*sizeof(float *),cudaMemcpyHostToDevice);
  float h_diff = n*n;
  unsigned int grid_rows = (n + blocksize - 1) / blocksize;
  dim3 DimBlock(blocksize,blocksize);
  dim3 DimGrid(grid_rows,grid_rows);
 printf(">> Num of Block = %d | Block Dim = %d |Matrix size = %d\n", grid_rows, blocksize, n);
  cudaMalloc(&d_diff, sizeof(float));
  cudaMemset(d_diff, 0, sizeof(float));
  int done = 0;
  while ((cnt_iter < MAX_ITER) && !done) {
    solver<<<DimGrid, DimBlock>>>(adi, ado, n, n, d_diff);
    cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_diff, 0, sizeof(float));
    printf("diff : %f \n ", h_diff);
    float **adt = adi; // ping-pong input and output buffers
    adi = ado;
    ado = adt;
    if (h_diff/n/n < TOL)
      done = 1;
    cnt_iter++;
    }
  printf("cnt_iter = %d, diff = %f\n", cnt_iter, h_diff/(n*n));
  for (int i = 0; i < n; i++){
    cudaMemcpy(a[i],temi[i],n*sizeof(float),cudaMemcpyDeviceToHost);
  }
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