#include <iostream>
#include <math.h>

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line)
{
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

__global__ void kernelA(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = 7;
}

__global__ void kernelB(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = blockIdx.x;
}

__global__ void kernelC(int *a) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  a[idx] = threadIdx.x;
}

void printArray(int *a, int N) {
    for (int i = 0; i < N; i++) {
      printf ("%*d ", 4, a[i]);
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
  if(argc != 4) {
    fprintf(stderr, "Usage: filename num_of_block block_dim array_length\n");
    return 1;
  }

  int nBlock = atoi(argv[1]);
  int blockDim = atoi(argv[2]);
  int N = atoi(argv[3]);
  printf(">> Num of Block = %d | Block Dim = %d | Length of Array = %d\n", nBlock, blockDim, N);

  // N is low number to make the result tractable to print
  // N = 128;
  int *x, *y, *z;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(int));
  cudaMallocManaged(&y, N*sizeof(int));
  cudaMallocManaged(&z, N*sizeof(int));

  // Run kernels on the GPU
  kernelA<<<nBlock, blockDim>>>(x);
  CHECK_LAST_CUDA_ERROR();
  kernelB<<<nBlock, blockDim>>>(y);
  CHECK_LAST_CUDA_ERROR();
  kernelC<<<nBlock, blockDim>>>(z);
  CHECK_LAST_CUDA_ERROR();

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Print the result
  std::cout << "Result of Kernel A" << std::endl;
  printArray(x, N);
  std::cout << "Result of Kernel B" << std::endl;
  printArray(y, N);
  std::cout << "Result of Kernel C" << std::endl;
  printArray(z, N);

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  
  std::cout << std::endl;
  return 0;
}
