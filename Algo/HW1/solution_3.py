def max_div3_sum(numbers: list) -> int:
    s = sum(numbers)
    if s % 3 == 0:
        return s
    
    r11, r12, r21, r22 = float('inf'), float('inf'), float('inf'), float('inf')
    
    for num in numbers:
        if num % 3 == 1:
            if num < r11:
                r12 = r11
                r11 = num
            elif num < r12:
                r12 = num
        elif num % 3 == 2:
            if num < r21:
                r22 = r21
                r21 = num
            elif num < r22:
                r22 = num
                
    if s % 3 == 1:
        options = []
        if r11 != float('inf'):
            options.append(s - r11)
        if r21 != float('inf') and r22 != float('inf'):
            options.append(s - r21 - r22)
        return max(options) if options else 0
    else:
        options = []
        if r21 != float('inf'):
            options.append(s - r21)
        if r11 != float('inf') and r12 != float('inf'):
            options.append(s - r11 - r12)
        return max(options) if options else 0

def solution():
    numbers = [int(x) for x in input().split()]
    result = max_div3_sum(numbers)
    print(result)

solution()