import random
import time

def generateRandomArray(size, maxVal):
    pass

def insertionSort(arr, left, right):
    pass

def merge():
    pass

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
    print("Time taken: " + duration)
    print("Number of comparisons: " + comparisons)


if __name__ == "main":
    main()


