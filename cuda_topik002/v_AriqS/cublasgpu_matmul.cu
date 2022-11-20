/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cublas_v2.h>
#define BLOCK_SIZE 2


/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < m && row < m) 
    {
        for(int i = 0; i < m; i++) 
        {
            sum += a[row * m + i] * b[i * m + col];
        }
        c[row * m + col] = sum;
    }
} 

__global__ void sequential_gpu_matrix_mult(int *d_a, int *d_b, int *d_result, int m) {
     for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < m; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < m; ++h) 
            {
                tmp += d_a[i * m + h] * d_b[h * m + j];
            }
            d_result[i * m + j] = tmp;
        }
    }
}

/*
*********************************************************************
function name: gpu_square_matrix_mult

description: dot product of two matrix (not only square) in GPU

parameters: 
            &a GPU device pointer to a n X n matrix (A)
            &b GPU device pointer to a n X n matrix (B)
            &c GPU device output purpose pointer to a n X n matrix (C) 
            to store the result
Note:
    grid and block should be configured as:

        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_square_matrix_mult(int *d_a, int *d_b, int *d_result, int n) 
{
    __shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub) 
    {
        idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
        if(idx >= n*n)
        {
            // n may not divisible by BLOCK_SIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
        if(idx >= n*n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }  
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}



/*
*********************************************************************
function name: gpu_matrix_transpose

description: matrix transpose

parameters: 
            &mat_in GPU device pointer to a rows X cols matrix
            &mat_out GPU device output purpose pointer to a cols X rows matrix 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dim_grid((n - 1) / BLOCK_SIZE + 1, (n - 1) / BLOCK_SIZE + 1, 1);
        dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

return: none
*********************************************************************
*/
__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int rows, unsigned int cols) 
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows) 
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}
/*
*********************************************************************
function name: gpu blas mmul

description: 

parameters: 
            
return: none
*********************************************************************
*/
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
     int lda=m,ldb=k,ldc=m;
     const float alpha = 1.0f;
     const float beta  = 0.0f;
 
     // Create a handle for CUBLAS
     cublasHandle_t handle;
     cublasCreate(&handle);

     // Do the actual multiplication
     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B, lda, A, ldb, &beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}

/*
*********************************************************************
function name: main

description: test and compare

parameters: 
           ./parallel_matmul.o matrix_size blockSize

return: none
compile :  nvcc cublasgpu_matmul.cu -lcublas -o cublasgpu_matmul.o

*********************************************************************
*/
int main(int argc, char** argv)
{
    int m;
    m= atoi(argv[1]);
    
    //int blockSize = atoi(argv[2]);
    
    /* Fixed seed for illustration */
    srand(3333);
    //printf("please type in m n and k\n");
    //scanf("%d %d %d", &m, &n, &k);

    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b, *h_c, *h_c_seq, *h_c_seq_transpose;
    cudaMallocHost((void **) &h_a, sizeof(float)*m*m);
    cudaMallocHost((void **) &h_b, sizeof(float)*m*m);
    cudaMallocHost((void **) &h_c, sizeof(float)*m*m);
    cudaMallocHost((void **) &h_c_seq, sizeof(float)*m*m);
    cudaMallocHost((void **) &h_c_seq_transpose, sizeof(float)*m*m);
    

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            h_a[i * m + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            h_b[i * m + j] = rand() % 1024;
        }
    }

    float gpu_elapsed_time_ms, seq_gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    // Allocate memory space on the device 
    float *d_a, *d_b, *d_c,*d_c_transpose ;
    cudaMalloc((void **) &d_a, sizeof(float)*m*m);
    cudaMalloc((void **) &d_b, sizeof(float)*m*m);
    cudaMalloc((void **) &d_c, sizeof(float)*m*m);
    cudaMalloc((void **) &d_c_transpose, sizeof(float)*m*m);
    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(float)*m*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*m*m, cudaMemcpyHostToDevice);

  
   // int nBlocks = m/blockSize + (m%blockSize == 0?0:1);
    //dim3 dimBlock(blockSize, blockSize);
    //dim3 dimGrid(nBlocks, nBlocks);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = grid_rows;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // Launch kernel 
    printf(">> Num of Block = %d | Block Dim = %d |Matrix size = %d\n", grid_rows, BLOCK_SIZE, m);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m);    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));
    // Transefr results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*m*m, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on non shared Parallel_GPU: %f ms.\n\n", m, m, m, m, gpu_elapsed_time_ms);

    // start the sequential version
    cudaEventRecord(start, 0);
    

    gpu_blas_mmul(d_a, d_b, d_c, m, m, m);   
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err2));
     // Transefr results from device to host 
    cudaMemcpy(h_c_seq, d_c, sizeof(int)*m*m, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&seq_gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on cublas Parallel_GPU: %f ms.\n\n", m, m, m, m, seq_gpu_elapsed_time_ms);

    // validate results computed by GPU
    //cudaMemcpy(d_c, h_c_seq, sizeof(float)*m*m, cudaMemcpyHostToDevice);
   /** gpu_matrix_transpose<<<dimGrid, dimBlock>>>(d_c, d_c_transpose, m,m);
    cudaError_t err3 = cudaGetLastError();
    if (err3 != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err3));
    cudaMemcpy(h_c_seq_transpose, d_c_transpose, sizeof(int)*m*m, cudaMemcpyDeviceToHost);*/


    int all_ok = 1;
    int error_counter = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_c_seq[i*m + j], i, j, h_c[i*m + j]);
            if((h_c_seq[i*m + j] != h_c[i*m + j] ) && error_counter<10)
            {
                error_counter++;
                printf("[%d][%d]:%f == [%d][%d]:%f, ", i, j, h_c_seq[i*m + j], i, j, h_c[i*m + j]);
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!!!, speedup = %f\n", seq_gpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("incorrect results\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_c_seq);
    return 0;
}