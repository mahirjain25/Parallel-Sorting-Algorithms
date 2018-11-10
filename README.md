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