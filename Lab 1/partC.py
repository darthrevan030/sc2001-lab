import random
import time
import matplotlib.pyplot as plt
import numpy as np


def generateRandomArray(size: int, maxVal: int) -> np.ndarray:
    return np.random.randint(1, maxVal + 1, size, dtype=np.int64)

# unused function
# def generateInputData(sizes: list[int], maxVal: int) -> list[np.ndarray]:
#     inputData = []
#     for size in sizes:
#         inputData.append(generateRandomArray(size, maxVal))
#     return inputData

def generateConsistentDatasets():
    print("Generating consistent datasets for all experiments...")
    
    # Set seed for reproducible datasets
    np.random.seed(42)
    random.seed(42)
    
    datasets = {}
    
    # Different sizes we'll test
    sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]
    maxVal = 10000000
    
    for size in sizes:
        datasets[size] = generateRandomArray(size, maxVal)
        print(f"  Generated dataset of size {size}")
    
    # Special dataset for analyzeFixedN (50000 elements)
    if 50000 not in datasets:
        datasets[50000] = generateRandomArray(50000, maxVal)
    
    # Special dataset for final comparison (10M elements)
    if 10000000 not in datasets:
        datasets[10000000] = generateRandomArray(10000000, maxVal)
    
    print("All datasets generated!\n")
    return datasets


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

def mergeSort(arr: np.ndarray, left: int, right: int) -> int:
    """Plain MergeSort implementation (baseline for comparison)."""
    comparisons = 0
    if left < right:
        mid = (left + right) // 2
        comparisons += mergeSort(arr, left, mid)
        comparisons += mergeSort(arr, mid + 1, right)
        comparisons += merge(arr, left, mid, right)
    return comparisons


def runExperiment(arr: np.ndarray, algorithm: str, s: int = None) -> tuple[int, float]:
    """Run sorting experiment and return (comparisons, time_taken)"""
    arrCopy = arr.copy()
    
    startTime = time.perf_counter()
    
    if algorithm == "hybrid":
        comparisonsMade = hybridSort(arrCopy, 0, len(arrCopy) - 1, s)
    elif algorithm == "mergesort":
        comparisonsMade = mergeSort(arrCopy, 0, len(arrCopy) - 1)
    else: 
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    endTime = time.perf_counter()
    
    return comparisonsMade, endTime - startTime

def analyzeFixedS(datasets):
    """Part (c)(i): Analyze with fixed S, varying input sizes"""
    print("=== Analysis with Fixed S (S=10), Varying Input Sizes ===")
    
    sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]
    sVal = 10
    
    sizeResults = []
    comparisonResults = []
    timeResults = []
    
    for size in sizes:
        print(f"Testing size: {size}")
        arr = datasets[size]
        
        comparisonsMade, timeTaken = runExperiment(arr, "hybrid", sVal)
        
        sizeResults.append(size)
        comparisonResults.append(comparisonsMade)
        timeResults.append(timeTaken)
        
        print(f"  Comparisons: {comparisonsMade}, Time: {timeTaken:.6f}s")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizeResults, comparisonResults, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)')
    plt.ylabel('Number of Comparisons')
    plt.title(f'Comparisons vs Input Size (S={sVal})')
    plt.grid(True, alpha=0.3)
    
    # Theoretical O(n log n) comparison
    theoretical = [n * np.log2(n) * 0.8 for n in sizeResults]
    plt.plot(sizeResults, theoretical, 'r--', linewidth=2, label='Theoretical O(n log n)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(sizeResults, timeResults, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (seconds)')
    plt.title(f'Time vs Input Size (S={sVal})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sizeResults, comparisonResults, timeResults

def analyzeFixedN(datasets):
    """Part (c)(ii): Analyze with fixed input size, varying S values"""
    print("=== Analysis with Fixed Input Size (n=50000), Varying S Values ===")
    
    nFixed = 50000
    sVal = list(range(1, 51, 2))  # S from 1 to 50, step 2
    maxVal = 1000000
    
    # Use the same pre-generated array for all S tests
    arr = datasets[nFixed]
    
    sResults = []
    comparisonResults = []
    timeResults = []
    
    for s in sVal:
        print(f"Testing S: {s}")
        
        comparisonsMade, timeTaken = runExperiment(arr, "hybrid", s)
        
        sResults.append(s)
        comparisonResults.append(comparisonsMade)
        timeResults.append(timeTaken)
        
        print(f"  Comparisons: {comparisonsMade}, Time: {timeTaken:.6f}s")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sResults, comparisonResults, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Threshold Value (S)')
    plt.ylabel('Number of Comparisons')
    plt.title(f'Comparisons vs Threshold S (n={nFixed})')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sResults, timeResults, 'go-', linewidth=2, markersize=6)
    plt.xlabel('Threshold Value (S)')
    plt.ylabel('Time (seconds)')
    plt.title(f'Time vs Threshold S (n={nFixed})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal S
    minTimeIndex = timeResults.index(min(timeResults))
    optimalS = sResults[minTimeIndex]
    print(f"Optimal S based on minimum time: {optimalS}")
    
    return sResults, comparisonResults, timeResults, optimalS

def findOptimalS(datasets):
    """Part (c)(iii): Find optimal S for different input sizes"""
    print("=== Finding Optimal S for Different Input Sizes ===")
    
    sizes = [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000, 10000000]
    sRange = list(range(5, 101, 5))  # S from 5 to 100, step 5
    maxVal = 1000000
    
    optimalSVal = []
    
    for size in sizes:
        print(f"Finding optimal S for size: {size}")
        arr = datasets[size]
        
        bestTime = float('inf')
        bestS = None
        
        for s in sRange:
            _, timeTaken = runExperiment(arr, "hybrid", s)
            
            if timeTaken < bestTime:
                bestTime = timeTaken
                bestS = s
        
        optimalSVal.append(bestS)
        print(f"  Optimal S: {bestS}, Best time: {bestTime:.6f}s")
    
    # Plot optimal S vs input size
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, optimalSVal, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Input Size (n)')
    plt.ylabel('Optimal Threshold (S)')
    plt.title('Optimal S vs Input Size')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return sizes, optimalSVal

def compareWithMergeSort(datasets, optimalS: int):
    """Part (d): Compare hybrid sort with original merge sort"""
    print("=== Comparison with Original Merge Sort (n=10M) ===")
    
    size = 10000000  # 10 million
    
    print(f"Using pre-generated array of size {size:,}")
    arr = datasets[size]
    
    # Test hybrid sort
    print("Running Hybrid Sort...")
    hybridComparisons, hybridTime = runExperiment(arr, "hybrid", optimalS)
    
    # Test original merge sort
    print("Running Original Merge Sort...")
    mergeComparisons, mergeTime = runExperiment(arr, "mergesort")
    
    # Results
    print("\n=== RESULTS ===")
    print(f"Hybrid Sort (S={optimalS}):")
    print(f"  Comparisons: {hybridComparisons:,}")
    print(f"  Time: {hybridTime:.6f} seconds")
    
    print(f"\nOriginal Merge Sort:")
    print(f"  Comparisons: {mergeComparisons:,}")
    print(f"  Time: {mergeTime:.6f} seconds")
    
    print(f"\nImprovement:")
    comparisonImprovement = ((mergeComparisons - hybridComparisons) / mergeComparisons) * 100
    timeImprovement = ((mergeTime - hybridTime) / mergeTime) * 100
    print(f"  Comparisons reduced by: {comparisonImprovement:.2f}%")
    print(f"  Time reduced by: {timeImprovement:.2f}%")
    
    return hybridComparisons, hybridTime, mergeComparisons, mergeTime

def main():
    # Generate input data
    datasets = generateConsistentDatasets()
    
    try:
        # Part (c)(i): Fixed S, varying input sizes
        print("\n1. Running analysis with fixed S...")
        analyzeFixedS(datasets)
        
        # Part (c)(ii): Fixed input size, varying S
        print("\n2. Running analysis with fixed input size...")
        _, _, _, optimalS = analyzeFixedN(datasets)
        
        # Part (c)(iii): Find optimal S for different sizes
        print("\n3. Finding optimal S for different input sizes...")
        findOptimalS(datasets)
        
        # Part (d): Compare with original merge sort
        print("\n4. Comparing with original merge sort...")
        compareWithMergeSort(datasets, optimalS)
        
        print("\nAll experiments completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
