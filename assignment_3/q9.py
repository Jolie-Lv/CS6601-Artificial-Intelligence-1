import numpy as np

def implies(a, b):
    if a is True and b is False: return False
    else: return True

def bicond(a, b):
    if a == b: return True
    else: return False

def table1(p1, p2):
    return !(p1 and p2)

def table2(p1, p2):
    (!p1) or p2


a = True
b = False
print(bicond(a, b))
