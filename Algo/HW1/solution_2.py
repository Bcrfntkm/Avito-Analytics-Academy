def max_even_sum(numbers: list) -> int:
    if sum(numbers) % 2:
        minimum = sum(numbers)
        for num in numbers:
            if num % 2 and num < minimum:
                minimum = num
        return sum(numbers) - minimum
    else:
        return sum(numbers)

def solution():
    numbers = [int(x) for x in input().split()]
    result = max_even_sum(numbers)
    print(result)

solution()