import random

lst = list(range(1, 50, 6))

print(lst)
print(lst.index(19, 0, 8))
random.shuffle(lst)
print(lst)
print(lst.index(19, 0, 8))