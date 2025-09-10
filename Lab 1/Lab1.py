import random
import time
import matplotlib.pyplot as plt
import numpy as np

def generateRandomArray(size: int, maxVal: int) -> list[int]:
    return [random.randint(1, maxVal) for i in range(size)]

def insertionSort(arr: list[int], left: int, right: int) -> None:
    # global comparisons variable -> track comparisons across the sorting algorithms
    global comparisons

    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i -1

        while j >= left:
            comparisons += 1

            if arr[j] <= key:
                break

            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key

def merge(arr: list[int], left: int, mid: int, right: int) -> None:
    # global comparisons variable -> track comparisons across the sorting algorithms
    global comparisons

    # create temporary arrays for left and right subarrays
    leftArr = arr[left:mid + 1]
    rightArr = arr[mid + 1: right + 1]

    # merge the temporary arrays back into arr[left...right]
    i = j = 0 # initialise indexes of the left and right subarrays
    k = left # initialise index of merged subarray

    while i < len(leftArr) and j < len(rightArr):
        comparisons += 1
        if leftArr[i] <= rightArr[j]:
            arr[k] = leftArr[i]
            i += 1
        else:
            arr[k] = rightArr[j]
            j += 1
        k += 1 

    # copy remaining elements of leftArr, if any
    while i < len(leftArr):
        arr[k] = leftArr[i]
        i += 1
        k += 1

    # copy remaining elements of rightArr, if any
    while j < len(rightArr):
        arr[k] = rightArr[j]
        j += 1
        k += 1

def hybridSort(arr, left, right, s):
    pass

def main():

    # global comparisons variable -> track comparisons across the sorting algorithms
    global comparisons

    # use system time for random seed to reduce chances of same random number being generated
    random.seed(int(time.time()))

    # get user input for size of array and maximum value in array
    size = int(input("Enter size of array: "))
    maxVal = int(input("Enter Maximum Value: "))

    # generate the array
    arr = generateRandomArray(size, maxVal)

    # optional - print the array
    print(arr)

    # start tracking time
    start = time.perf_counter()

    # call hybrid sort algorithm
    hybridSort(arr, 0, len(arr)-1, 2)

    # stop tracking time
    end = time.perf_counter()

    # calculate time taken
    duration = end - start

    # print results
    print("Time taken: ", duration)
    print("Number of comparisons: ", comparisons)


if __name__ == "main":
    main()


