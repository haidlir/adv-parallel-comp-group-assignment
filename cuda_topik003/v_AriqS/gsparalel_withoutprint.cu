#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAX 100
#define MAX_ITER 13
#define TOL 0.000001

// Generate a random float number with the maximum value of max

void print_matrix(float * N, int rows, int columns,int n) {

  int i, j;
  //print only 4*4 matrix.
  if (rows > 4)
    rows = 4;
  if (columns > 4)
    columns = 4;
  printf("Printing only first 4 results \n");
  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns; j++) {
     
        printf("%lf \t", N[((n * i) + j)]); //row major accessing with red color.
      

    }
    
  }
  
}

float rand_float(int max){
  return ((float)rand()/(float)(RAND_MAX)) * max;
}


// Allocate 2D matrix




// solver




__global__ void parallel_solver(float *mat, int n, int m){
  float diff = 0, temp;
	int done = 0, cnt_iter = 0, myrank;
    int index = threadIdx.x;
    int blockdim = blockDim.x;
    int numperthread = n*n / blockdim;
    
     int startpoint,endpoint;
    if(index == 0){
        startpoint =  numperthread * index + n ;
    }else{
        startpoint = numperthread * index;
    }

    
   if(index == (blockdim -1)){
   // printf("last thread ");
    endpoint = (n*n) - n;
   }else{
   endpoint = numperthread * (index + 1) - 1;
   }
   //printf(" number elements per thread = %d , thread index : %d , start at %d, end at %d \n",numperthread, index, startpoint , endpoint);
  	while (!done && (cnt_iter < MAX_ITER)) {
  		diff = 0;

  		// Neither the first row nor the last row are solved
  		// (that's why it starts at "n" and it goes up to "num_elems - 2n")
  		for (int i = startpoint; i < endpoint ; i++) {

  			// Additionally, neither the first nor last column are solved
  			// (that's why the first and last positions of "rows" are skipped)
  			if ((i % n == 0) || (i+1 % n == 0)) {
				continue;
			}

  			int pos_up = i - n;
  			int pos_do = i + n;
  			int pos_le = i - 1;
  			int pos_ri = i + 1;

  			temp = mat[i];
			mat[i] = 0.2 * (mat[i] + mat[pos_le] + mat[pos_up] + mat[pos_ri] + mat[pos_do]);
			diff += abs(mat[i] - temp);
           
      	}
		
		// printf("iteration %d diff at thread %d : %f \n",cnt_iter, threadIdx.x, diff);
      
  //print only 4*4 matrix.
  /*
        printf("Printing only first 4 results \n");
        for (x = 0; x < 4; x++) {
             printf("\n");
             for (y = 0; y < 4; y++) {
                
         printf("%lf \t", mat[((n * x) + y)]); //row major accessing with red color.
         }
        }*/
		if (diff/n/n < TOL) {
			done = 1;
			//printf("diff : %f \n",diff);
			//print_matrix(*mat,n,n,n);
			
		}
		cnt_iter ++;
	}


	if (done) {
		//printf("Solver converged after %d iterations\n", cnt_iter);

	}
	else {
	//	printf(" Solver not converged after %d iterations\n", cnt_iter);
         /*printf("Printing only first 4 results \n");
           int x, y;
        for (x = 0; x < 4; x++) {
             printf("\n");
             for (y = 0; y < 4; y++) {
                
         printf("%lf \t", mat[((n * x) + y)]); //row major accessing with red color.
         }
        }*/
	}
  }

  /*
__global__ void sequential_solver(float *mat, int n, int m){
  float diff = 0, temp;
  int done = 0, cnt_iter = 0, i, j;

  while (!done && (cnt_iter < MAX_ITER)){
    diff = 0;
    for (i = 1; i < n - 1; i++)
      for (j = 1; j < m - 1; j++){
	temp = mat[i][j];
	mat[i][j] = 0.2 * (mat[i][j] + mat[i][j - 1] + mat[i - 1][j] + mat[i][j + 1] + mat[i + 1][j]);
	diff += abs(mat[i][j] - temp);
      }
    if (diff/n/n < TOL)
      done = 1; 
    printf("diff : %f \n",diff);
    cnt_iter ++;
  }


  if (done){
    printf("Solver converged after %d iterations\n", cnt_iter);
    printf("final matrix:\n");
  
  
	 for (i = 0; i < 4; i++) {
        printf("\n");
        for (j = 0; j < 4; j++) {
       
       printf("%f \t", mat[i][j]); //row major accessing with red color.
        }
    }
    }
  else{printf("Solver not converged after %d iterations\n", cnt_iter);}
  }
*/



int main(int argc, char *argv[]) {

  int n;
  float *ahost;
  if (argc < 2) {
    printf("Call this program with two parameters: matrix_size communication \n");
    printf("\t matrix_size: Add 2 to a power of 2 (e.g. : 18, 1026)\n");
    exit(1);
    }

  n = atoi(argv[1]);
  int blocksize = atoi(argv[2]);
 // float **temp;
  printf("Matrix size = %d\n", n);
  cudaMallocHost((void **) &ahost, sizeof(float)*n*n);
  
  for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            ahost[i * n + j] = rand_float(MAX);
        }
    }

     int x, y;
  //print only 4*4 matrix.
        printf("Printing only first 4 results \n");
        for (x = 0; x < 4; x++) {
            printf("\n");
             for (y = 0; y < 4; y++) {
                
         printf("%lf \t", ahost[((n * x) + y)]); //row major accessing with red color.
         }
        }

 
// Allocate 2D array in Device
     float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
float *adevice;
  cudaMalloc((void **) &adevice, sizeof(float)*n*n);
 cudaMemcpy(adevice, ahost, sizeof(float)*n*n, cudaMemcpyHostToDevice);
  
  
    printf(">> Num of Block = 1 | Block Dim = 1 |Matrix size = %d\n", n);
    parallel_solver<<<1, blocksize>>>(adevice, n, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
   /* for (int i = 0; i < n; i++){
    cudaMemcpy(a[i],adi[i],n*sizeof(float),cudaMemcpyDeviceToHost);
  }*/
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
   
    printf("Time elapsed on GPU: %f ms.\n\n",  gpu_elapsed_time_ms);
    
  return 0;
}