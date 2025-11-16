import heapq

class StreamMedian:
    def __init__(self):
        self.lo = []   
        self.hi = []   

    def add_num(self, num: int) -> None:
        if not self.lo or num <= -self.lo[0]:
            heapq.heappush(self.lo, -num)
        else:
            heapq.heappush(self.hi, num)
        if len(self.lo) > len(self.hi) + 1:
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
        elif len(self.hi) > len(self.lo) + 1:
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self) -> float:
        if not self.lo and not self.hi:
            return None
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        if len(self.hi) > len(self.lo):
            return self.hi[0]
        return (-self.lo[0] + self.hi[0]) / 2


def solution():
    n = int(input())
    stream = StreamMedian()
    for i in range(n):
        line = input().split()
        command = line[0]
        if command == "ADD":
            stream.add_num(int(line[1]))
        elif command == "FIND_MEDIAN":
            print(f'{stream.find_median():.1f}')


solution()