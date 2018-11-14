#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 256 // 2^7
#define BLOCKS 1024 // 2^10
#define NUM_VALS THREADS*BLOCKS
#define MAX_DEPTH       16
#define INSERTION_SORT  32


#define swap(A,B) { int temp = A; A = B; B = temp;}

typedef struct vars{
  int l;
  int r;
  int leq;
} vars;

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__device__ void selection_sort( int *data, int left, int right )
{
  for( int i = left ; i <= right ; ++i ){
    int min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for( int j = i+1 ; j <= right ; ++j ){
      int val_j = data[j];
      if( val_j < min_val ){
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if( i != min_idx ){
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}

__global__ void cdp_simple_quicksort(int *data, int left, int right, int depth ){
    //If we're too deep or there are few elements left, we use an insertion sort...
    if( depth >= MAX_DEPTH || right-left <= INSERTION_SORT ){
        selection_sort( data, left, right );
        return;
    }

    cudaStream_t s,s1;
    int *lptr = data+left;
    int *rptr = data+right;
    int  pivot = data[(left+right)/2];

    int lval;
    int rval;

    int nright, nleft;

    // Do the partitioning.
    while (lptr <= rptr){
        // Find the next left- and right-hand values to swap
        lval = *lptr;
        rval = *rptr;

        // Move the left pointer as long as the pointed element is smaller than the pivot.
        while (lval < pivot && lptr < data+right){
            lptr++;
            lval = *lptr;
        }

        // Move the right pointer as long as the pointed element is larger than the pivot.
        while (rval > pivot && rptr > data+left){
            rptr--;
            rval = *rptr;
        }

        // If the swap points are valid, do the swap!
        if (lptr <= rptr){
            *lptr = rval;
            *rptr = lval;
            lptr++;
            rptr--;
        }
    }

    // Now the recursive part
    nright = rptr - data;
    nleft  = lptr - data;

    // Launch a new block to sort the left part.
    if (left < (rptr-data)){
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s >>>(data, left, nright, depth+1);
        cudaStreamDestroy(s);
    }

    // Launch a new block to sort the right part.
    if ((lptr-data) < right){
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cdp_simple_quicksort<<< 1, 1, 0, s1 >>>(data, nleft, right, depth+1);
        cudaStreamDestroy(s1);
    }
}


// gcc compiled code will call this function to access CUDA Quick Sort.
// This calls the kernel, which is recursive. Waits for it, then copies it's
// output back to CPU readable memory.
extern "C"
void gpu_qsort(int *data, int n){
    int* gpuData;
    int left = 0;
    int right = n-1;

    // Prepare CDP for the max depth 'MAX_DEPTH'.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    // Allocate GPU memory.
    cudaMalloc((void**)&gpuData,n*sizeof(int));
    cudaMemcpy(gpuData,data, n*sizeof(int), cudaMemcpyHostToDevice);

    // Launch on device
    cdp_simple_quicksort<<< 1, 1 >>>(gpuData, left, right, 0);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(data,gpuData, n*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuData);
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}


int main(int argc, char const *argv[])
{
  clock_t start, stop;

  int *values = (int*)malloc(NUM_VALS * sizeof(int));

  FILE *f = fopen("reverse_dataset.txt", "r");

  for(int i=0;i< NUM_VALS; i++) {
    fscanf(f, "%d\n", &values[i]);
  }

  printf("Hello\n");
  start  = clock();
  gpu_qsort(values, NUM_VALS);
  stop = clock();

  print_elapsed(start, stop);
  return 0;
}
