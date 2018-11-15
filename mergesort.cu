#include <stdlib.h>
#include <stdio.h>
#include <time.h>



#define THREADS 128 // 2^7
#define BLOCKS 1024 // 2^10
#define NUM_VALS THREADS*BLOCKS

#define checkCudaErrors(func)                                                \
{                                                                       \
  cudaError_t E  = func;                                                \
  if(E != cudaSuccess)                                                  \
  {                                                                     \
    printf( "\nError at line: %d ", __LINE__);                          \
    printf( "\nError:  %s ", cudaGetErrorString(E));                    \
  }                                                                     \
}  

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}


//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//

void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    // Actually allocate the two arrays
    checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(long)));


    // Copy from our input list into the first array
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));

    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

 
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));



    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;


    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;


        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }


    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));

    
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
   
}




int main(int argc, char const *argv[])
{
  clock_t start, stop;

  long *values = (long*)malloc(NUM_VALS * sizeof(long));

  FILE *f = fopen("reverse_dataset.txt", "r");

  for(int i=0;i< NUM_VALS; i++) {
    fscanf(f, "%d\n", &values[i]);
  }

  dim3 blocks(BLOCKS,1);    
  dim3 threads(THREADS,1); 
  start  = clock();
  mergesort(values, NUM_VALS, threads, blocks);
  stop = clock();

  print_elapsed(start, stop);
  return 0;
}
