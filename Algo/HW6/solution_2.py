import heapq

def merge_k_sorted(arrs: list) -> list:
    heap = []
    result = []

    for i, arr in enumerate(arrs):
        heapq.heappush(heap, (arr[0], i, 0))

    while heap:
        val, arr_i, el_i = heapq.heappop(heap)
        result.append(val)
        next_i = el_i + 1
        if next_i < len(arrs[arr_i]):
            heapq.heappush(heap, (arrs[arr_i][next_i], arr_i, next_i))

    return result


def solution():
    arrs = read_multiline_input() # эта функция уже написана
    merged = merge_k_sorted(arrs)
    print(' '.join([str(el) for el in merged]))

solution()