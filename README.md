# Parallel-Sorting-Algorithms
Comparison of several parallel sorting algorithms, using the Nvidia CUDA programming framework.


### Instructions to generate the inputs

##### Random Input

Run `g++ random_generator.cpp -o generate` followed by `./generate`.

The numbers will be stored in a file titled **random_dataset.txt**


##### Reverse Sorted Input

Run `g++ reverse_sorted_generator.cpp -o generate_reverse` followed by `./generate_reverse`.

The numbers will be stored in a file titled **reverse_dataset.txt**

##### Sorted Input

Run `g++ sorted_generator.cpp -o generate_sorted` followed by `./generate_sorted`.

The numbers will be stored in a file titled **sorted_dataset.txt**

##### Almost Sorted Input

**Note: The approach we used for this was to randomly carry out swaps after sorting the vector. Number of swaps = 0.1 x Number of Total Elements. This way atmost 20% of elements are out of place**

Run `g++ almost_sorted_generator.cpp -o generate_sorted` followed by `./generate_almost_sorted`.

The numbers will be stored in a file titled **almost_sorted_dataset.txt**


## Sorting Algorithms

### Bitonic Sort
To generate the executable, run 
``` nvcc bitonic_sort.cu ```

to run the executable, run
```./a.out ```

### Merge Sort
To generate the executable, run 
```nvcc -o mergesort.cu ```

to run the executable, run
```./a.out ```

### Quick Sort
**important** - since our QuickSort() implementation uses a recursive function on the device, we need to specify our architecture as *sm_35*
To generate the executable, run 
```nvcc -arch=sm_35 -rdc=true quick.cu  ```

to run the executable, run
```./a.out ```