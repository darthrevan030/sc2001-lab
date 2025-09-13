import random
import time
import matplotlib.pyplot as plt
import numpy as np


comparisons = 0

def generateRandomArray(size: int, maxVal: int) -> np.ndarray:
    return np.random.randint(1, maxVal + 1, size, dtype=np.int64)

def generateInputData(sizes: list[int], maxVal: int) -> list[np.ndarray]:
    inputData = []
    for size in sizes:
        inputData.append(generateRandomArray(size, maxVal))
    return inputData

def insertionSort(arr: np.ndarray, left: int, right: int) -> int:
    comparisons = 0
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left:
            comparisons += 1
            if arr[j] <= key:
                break
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return comparisons

def merge(arr: np.ndarray, left: int, mid: int, right: int) -> int:
    comparisons = 0
    leftArr = arr[left:mid + 1].copy()
    rightArr = arr[mid + 1: right + 1].copy()
    i = j = 0
    k = left
    while i < len(leftArr) and j < len(rightArr):
        comparisons += 1
        if leftArr[i] <= rightArr[j]:
            arr[k] = leftArr[i]
            i += 1
        else:
            arr[k] = rightArr[j]
            j += 1
        k += 1
    while i < len(leftArr):
        arr[k] = leftArr[i]
        i += 1
        k += 1
    while j < len(rightArr):
        arr[k] = rightArr[j]
        j += 1
        k += 1
    return comparisons

def hybridSort(arr: np.ndarray, left: int, right: int, s: int) -> int:
    comparisons = 0
    if left < right:
        if right - left + 1 <= s:
            comparisons += insertionSort(arr, left, right)
        else:
            mid = (left + right) // 2
            comparisons += hybridSort(arr, left, mid, s)
            comparisons += hybridSort(arr, mid + 1, right, s)
            comparisons += merge(arr, left, mid, right)
    return comparisons

def main():

    # global comparisons variable -> track comparisons across the sorting algorithms
    global comparisons

    # use system time for random seed to reduce chances of same random number being generated
    random.seed(int(time.time()))

    # Define sizes for plotting (logarithmic scale, clean output)
    sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    maxVal = 100000000000

    # Generate input data
    input_data = generateInputData(sizes, maxVal)

    times = []
    comparisons_list = []

    for arr in input_data:
        arr_copy = arr.copy()
        start = time.perf_counter()
        comparisons = hybridSort(arr_copy, 0, len(arr_copy)-1, 10)
        end = time.perf_counter()
        duration = end - start
        times.append(duration)
        comparisons_list.append(comparisons)
        print(f"Size: {len(arr_copy)}, Time: {duration:.6f} s, Comparisons: {comparisons}")

    # Plotting
    plt.figure(figsize=(10,6))
    plt.plot(sizes, times, marker='o')
    plt.xscale('log')
    plt.xlabel('Input Size (log scale)')
    plt.ylabel('Time (seconds)')
    plt.title('Hybrid Sort Time Complexity')
    plt.grid(True)
    plt.show()

    # Optionally, plot comparisons as well
    plt.figure(figsize=(10,6))
    plt.plot(sizes, comparisons_list, marker='o', color='orange')
    plt.xscale('log')
    plt.xlabel('Input Size (log scale)')
    plt.ylabel('Comparisons')
    plt.title('Hybrid Sort Comparisons')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()


