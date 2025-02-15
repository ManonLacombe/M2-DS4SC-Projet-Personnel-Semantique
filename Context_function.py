"""
  Original CED Source by Clément Moreau
"""

from math import *

"""
Context function f_k
:param k: Int -- Index of edition 
:param x: Float -- Antecedent
:param a: Float -- Standard Deviation
"""

def unit(k: int, x: float, sigma: float) -> float:
    return 1.

def gaussian(k: int, x: float, sigma: float = 15.0)-> float:
    return exp(-1.0 / 2 * ((x - k) / (sigma/2))**2)

def prep_gaussian(sigma: float):
    def internal(k, x):
        return exp(-1.0 / 2 * ((x - k) / (sigma/2))**2)
    return internal

def gaussianOlap(k: int, x: float, L)-> float:
    return exp(-1.0 / 2 * (2*(sqrt((k+1))*(x - k)/L))**2)


def fuzzy(a, b, c, d, x):
    if(x >= a and x < b) :
        return 1.0 / (b-a) * x + a / (a-b)
    elif(x >= b and x < c) :
        return 1
    elif(x >= c and x < d) :
        return 1.0 / (c-d) * x + d / (d-c)
    else :
        return 0
