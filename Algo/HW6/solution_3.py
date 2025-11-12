import bisect


class StreamMedian:
    def __init__(self):
        self.arr = [float('inf')]
        self.n = 0

    def add_num(self, num: int) -> None:
        pos = bisect.bisect_left(self.arr, num) # вроде как так можно)))
        self.arr.insert(pos, num)
        self.n += 1

    def find_median(self) -> float:
        if self.n % 2:
            return self.arr[self.n // 2]
        else:
            return (self.arr[self.n // 2] + self.arr[self.n // 2 - 1]) / 2


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