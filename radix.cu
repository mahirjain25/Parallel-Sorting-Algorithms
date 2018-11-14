#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
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
  printf("Elapsed time: %fs\n", elapsed);
}


int main(int argc, char const *argv[])
{
  clock_t start, stop;

  int *val = (int*)malloc(NUM_VALS * sizeof(int));
  thrust::device_vector<int> values(NUM_VALS);

  FILE *f = fopen("random_dataset.txt", "r");

  for(int i=0;i< NUM_VALS; i++) {
    fscanf(f, "%d\n", &val[i]);
  }
  for(int i=0;i< NUM_VALS; i++) {
    values[i] = val[i];
  }



  printf("Hello\n");
  cudaEvent_t estart, estop;
    cudaEventCreate( &estart );
    cudaEventCreate( &estop );
  start  = clock();
  cudaEventRecord( estart, 0 );
  thrust::sort(values.begin(), values.end());
  cudaEventRecord( estop, 0 ) ;
  cudaEventSynchronize( estop );
  stop = clock();
  float elapsedTime;
    cudaEventElapsedTime( &elapsedTime,
        estart, estop ) ;
  for(int i=0;i< NUM_VALS; i++) {
    val[i] = values[i];
  }

  for (int i = 0; i < 20; ++i)
  {
    printf("%d\n", val[i]);
  }

  printf("Elapsed time: %f\n", elapsedTime);
  return 0;
}
