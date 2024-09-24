def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes_composites(n):
    primes = set()
    composites = set()
    for i in range(2, n + 1):
        if is_prime(i):
            primes.add(i)
        else:
            composites.add(i)
    return primes, composites

def is_good_number(n):
    primes, composites = get_primes_composites(n)
    return len(primes) == len(composites)

good_numbers_sum = 0
n = 2

while True:
    if is_good_number(n):
        good_numbers_sum += n
        primes, composites = get_primes_composites(n)
        print(f"好数: {n}")
        print(f"  质数集合: {primes}")
        print(f"  合数集合: {composites}")
        print(f"  当前好数之和: {good_numbers_sum}")
        print()
    if good_numbers_sum in [33, 34, 2013, 2014]:
        print("good number:", n)
        print("good number sum :", good_numbers_sum)
        break
    n += 1

print(f"所有好数之和为: {good_numbers_sum}")
if good_numbers_sum == 33:
    print("答案是 A")
elif good_numbers_sum == 34:
    print("答案是 B")
elif good_numbers_sum == 2013:
    print("答案是 C")
elif good_numbers_sum == 2014:
    print("答案是 D")
else:
    print("没有匹配的答案")