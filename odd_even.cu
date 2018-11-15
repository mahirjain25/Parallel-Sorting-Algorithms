#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 128 // 2^7
#define BLOCKS 1024 // 2^10
#define NUM_VALS THREADS*BLOCKS

#define swap(A,B) { int temp = A; A = B; B = temp;}

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__global__ void odd_even_sort(int *c,int *count)
{
    int l;
    if(*count%2==0)
          l=*count/2;
    else
         l=(*count/2)+1;
    for(int i=0;i<l;i++)
    {
            if((!(threadIdx.x&1)) && (threadIdx.x<(*count-1)))  //even phase
            {
                if(c[threadIdx.x]>c[threadIdx.x+1])
                  swap(c[threadIdx.x], c[threadIdx.x+1]);
            }

            __syncthreads();
            if((threadIdx.x&1) && (threadIdx.x<(*count-1)))     //odd phase
            {
                if(c[threadIdx.x]>c[threadIdx.x+1])
                  swap(c[threadIdx.x], c[threadIdx.x+1]);
            }
            __syncthreads();
    }

}

void odd_even_caller(int *values)
{
  int *dev_values, *count;
  size_t size = NUM_VALS * sizeof(int);
  int n = NUM_VALS;

  cudaMalloc((void**) &dev_values, size);
  cudaMalloc((void**)&count,sizeof(int));
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaMemcpy(count,&n,sizeof(int),cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    
  dim3 threads(THREADS,1);  

  odd_even_sort<<<blocks, threads>>>(dev_values, count);
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
  odd_even_caller(values);
  for(int i=0; i < 20 ;i ++) {
    printf("%d\n", values[i]);
  }
  stop = clock();

  print_elapsed(start, stop);
  return 0;
}
