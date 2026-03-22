import heapq

def get_kth_element(arr: list, k: int):
    heap = [-x for x in arr[:k + 1]]
    heapq.heapify(heap)  
    for x in arr[k + 1:]:
        if x < -heap[0]:
            heapq.heapreplace(heap, -x)
    #print(heap)
    return -heap[0]


def solution():
    arr = list(map(int, input().split()))
    k = int(input())
    print(get_kth_element(arr, k))

solution()