#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 128 // 2^7
#define BLOCKS 1024 // 2^10
#define NUM_VALS THREADS*BLOCKS

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__global__ void bitonic_sort_step(int *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        // swap
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        // swap
        int temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

void bitonic_sort(int *values)
{
  int *dev_values;
  size_t size = NUM_VALS * sizeof(int);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    
  dim3 threads(THREADS,1);  

  int j, k;
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
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
  bitonic_sort(values);
  for(int i=0; i < 20 ;i ++) {
    printf("%d\n", values[i]);
  }
  stop = clock();

  print_elapsed(start, stop);
  return 0;
}
