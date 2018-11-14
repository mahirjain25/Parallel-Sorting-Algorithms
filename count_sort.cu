#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 128 // 2^7
#define BLOCKS 1024 // 2^10
#define NUM_VALS THREADS*BLOCKS
#define MAX_VALUE 8196

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__global__ void count(int *A, int *B, int n) {

	int b_id 		= blockIdx.x,
			b_num 	= gridDim.x,
			b_size,
			b_offset,
			t_id 	= threadIdx.x,
			t_num 	= blockDim.x,
			t_size,
			t_offset,
			offset;

	// initialize a shared memory array to store the count for each block.
	__shared__ int count[MAX_VALUE];

	// set intial values to zeros. Each thread sets its own share to zero.
	t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
	offset = t_id * t_size;
	for (int i = offset; i < offset + t_size && i < MAX_VALUE; ++i)
		count[i] = 0;

	// wait until all threads have completed the initialization process.
	__syncthreads();

	// accumulate the counts of each value. Each thread counts a certain portain
	// of the unsorted array.
	b_size = (b_num > n ? 1 : n / b_num);
	b_offset = b_id * b_size;

	t_size = (t_num > b_size ? 1 : b_size / t_num);

	offset = b_offset + t_id * t_size;
	for (int i = offset; i < offset + t_size && i < b_offset + b_size && i < n; ++i)
		atomicAdd(&count[A[i]], 1);

	// wait until all threads have completed the couting phase.
	__syncthreads();

	// copy the block count into global memory. Each thread copies its portioin to 
	// the global memory.
	t_size = (t_num > MAX_VALUE ? 1 : MAX_VALUE / t_num);
	t_offset = t_id * t_size;
	offset = b_id * MAX_VALUE + t_offset;

	if (offset + t_size <= (b_id + 1) * MAX_VALUE)
		memcpy(&B[offset], &count[t_offset], sizeof(int) * t_size);

}

__global__ void merge(int *B) {

	int b_id	= blockIdx.x,
		b_num	= gridDim.x,
		b_size,
		b_offset,
		t_id	= threadIdx.x,
		t_num	= blockDim.x,
		t_size,
		offset;

	// loop through and merge until all arrays are merged.
	for (int i = b_num, j = 2; i != 1; i /= 2, j *= 2) {

		// each block will operate on b_size values which equal, the number of 
		// count arrays * size of count arrays / number of blocks / 2. The final 2
		// represents the merge process.
		b_size = i * MAX_VALUE / b_num / 2;
		b_offset = (b_id / j) * (j * MAX_VALUE) + b_size * (b_id % j);

		t_size = (t_num > b_size ? 1 : b_size / t_num);

		// calculate the offset that each thread will start at and sum counts.
		offset = b_offset + t_id * t_size;
		for (int k = offset, l = offset + (MAX_VALUE * (j / 2)); 
			k < offset + t_size && k < b_offset + b_size; ++k, ++l)
			B[k] += B[l];

		// wait untill all arrays are merged for every step.
		__syncthreads();

	}

}

void bitonic_sort(int *values)
{
    int *dev_values, *dev_ans;
    size_t size = NUM_VALS * sizeof(int);

    size_t size1 = MAX_VALUE * BLOCKS * sizeof(int);

    cudaMalloc((void**) &dev_values, size);
    cudaMalloc((void**) &dev_ans, size1);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS,1);    
    dim3 threads(THREADS,1);  

    count<<<blocks, threads>>>(dev_values, dev_ans, NUM_VALS);
    merge<<<blocks, threads>>>(dev_ans);

    int ans[NUM_VALS];
    cudaMemcpy(ans, dev_ans, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);

    // Construct sorted array
    for (int i = 0, j = 0; i < MAX_VALUE; ++i) {
		for (int k = 0; k < ans[i]; ++k, ++j) {
				values[j] = i;
		}
	}
}

int main(int argc, char const *argv[])
{
  clock_t start, stop;

  int *values = (int*)malloc(NUM_VALS * sizeof(int));

  FILE *f = fopen("random_dataset.txt", "r");

  for(int i=0;i< NUM_VALS; i++) {
    fscanf(f, "%d\n", &values[i]);
  }
      cudaEvent_t estart, estop;
    
    

  start  = clock();
  cudaEventCreate( &estart );
  cudaEventRecord( estart, 0 );
  bitonic_sort(values);
  cudaEventCreate( &estop );
  cudaEventRecord( estop, 0 ) ;
  cudaEventSynchronize( estop );
  stop = clock();
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime,
        estart, estop ) ;

  printf("Elapsed time: %f\n", elapsedTime);
  //print_elapsed(start, stop);
  return 0;
}
