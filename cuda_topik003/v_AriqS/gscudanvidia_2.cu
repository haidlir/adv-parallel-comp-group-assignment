#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX_ITER 1000000
#define MAX 100 //maximum value of the matrix element
#define TOL 0.000001

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
__global__ void solver(float **matd, int n, int m, bool debug){
  float diff = 0, temp;
  int done = 0, cnt_iter = 0;
 int j= blockIdx.x*blockDim.x + threadIdx.x;
 int i= blockIdx.y*blockDim.y + threadIdx.y;
 if (debug) printf("valor de matd:%f\n",**matd);
  while (!done && (cnt_iter < MAX_ITER)){
    diff = 0;
      if ((i < n - 1) && (j < m - 1) && (i > 0) && (j > 0)){
       temp = (matd)[i][j];
        (matd)[i][j] = 0.2 * ((matd)[i][j] + (matd)[i][j - 1] + (matd)[i - 1][j] + (matd)[i][j + 1] + (matd)[i + 1][j]);
        diff += abs((matd)[i][j] - temp);
  //      printf("diff:%f\n",diff);
      }

    if (diff/n/n < TOL)
      done = 1;
    cnt_iter ++;
  }
  if (debug){
    if (done)
      printf("Solver converged after %d iterations\n", cnt_iter);
    else
      printf("Solver not converged after %d iterations\n", cnt_iter);
  }
}

int main(int argc, char *argv[]) {
  int n;
  float **a,**ad;
struct timeval start, end,start1, end1;
    double mtime, seconds, useconds,x,mtime1, seconds1, useconds1,y;
    gettimeofday(&start, NULL);
        dim3 DimGrid(10);
        dim3 DimBlock(128);
  if (argc < 2) {
    printf("Call this program with two parameters: matrix_size communication \n");
    printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");

    exit(1);
  }

  n = atoi(argv[1]);
  float *temph[n];
  printf("Matrix size = %d\n", n);
  allocate_init_2Dmatrix(&a, n, n);

// Allocate 2D array in Device
  cudaMalloc((void **)&ad,n*sizeof(float *));
 for (int i = 0; i < n; i++){
        cudaMalloc(&temph[i], n*sizeof(float));
}
  cudaMemcpy(ad,temph,n*sizeof(float *),cudaMemcpyHostToDevice);

 for (int i = 0; i < n; i++){
  cudaMemcpy(temph[i],a[i],n*sizeof(float),cudaMemcpyHostToDevice);
}

}