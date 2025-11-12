def is_palindrome(a: int) -> bool:
    return str(a) == str(a)[::-1]


def solution():
    a = int(input())
    c = is_palindrome(a)
    print(c)

solution()