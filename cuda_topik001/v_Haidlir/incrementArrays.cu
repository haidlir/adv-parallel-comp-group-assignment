#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line) {
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
              << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
    // We don't exit when we encounter CUDA errors in this example.
    // std::exit(EXIT_FAILURE);
  }
}

void incrementArrayOnHost(float *a, int N) {
  int i;
  for (i=0; i < N; i++) {
    a[i] = a[i]+1.f;
  }
}

__global__ void incrementArrayOnDevice(float *a, int N) {
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx<N) {
    a[idx] = a[idx]+1.f;
  }
}

int main(int argc, char** argv) {
  if(argc != 3) {
    fprintf(stderr, "Usage: filename blockSize N\n");
    return 1;
  }

  int blockSize = atoi(argv[1]);
  int N = atoi(argv[2]);

  float *a_h, *b_h; // pointers to host memory
  float *a_d; // pointer to device memory
  int i;
  size_t size = N*sizeof(float);

  // allocate arrays on host
  a_h = (float *)malloc(size);
  b_h = (float *)malloc(size);

  // allocate array on device
  cudaMalloc((void **) &a_d, size);

  // initialization of host data
  for (i=0; i<N; i++) a_h[i] = 1.0f;

  // copy data from host to device
  cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);

  // do calculation on host
  incrementArrayOnHost(a_h, N);

  // do calculation on device:
  // Part 1 of 2. Compute execution configuration
  int nBlocks = N/blockSize + (N%blockSize == 0?0:1);
  printf(">> Num of Block = %d | Block Dim = %d | Length of Array = %d\n", nBlocks, blockSize, N);

  // Part 2 of 2. Call incrementArrayOnDevice kernel
  incrementArrayOnDevice <<< nBlocks, blockSize >>> (a_d, N);
  CHECK_LAST_CUDA_ERROR();
  cudaDeviceSynchronize();

  // Retrieve result from device and store in b_h
  cudaMemcpy(b_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

  // check results
  for (i=0; i<N; i++) assert(a_h[i] == b_h[i]);

  // cleanup
  free(a_h); free(b_h); cudaFree(a_d);

  std::cout << "OK!" << std::endl;
  return 0;
}