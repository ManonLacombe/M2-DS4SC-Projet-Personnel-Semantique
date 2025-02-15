"""
  Original CED Source by Clément Moreau
"""

from enum import Enum
from typing import TypeVar
from numba import jit
import numba
import numpy as np

ADD: int = 1
MOD: int = 2
DEL: int = 3


T = TypeVar('T')

class Operator :
    def __init__(self, op: int, S, a: T, k: int) :
        """
        :param op: Contextual Edit Operation name
        :param S: NumpyArray<T> --  Edited Sequence
        :param a: T -- Symbol to edit
        :param k: Int -- Index of edition
        """
        self.op = op
        self.S = S
        self.a = a
        self.k = k