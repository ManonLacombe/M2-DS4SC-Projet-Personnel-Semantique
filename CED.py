"""
  Original CED Source by Clément Moreau
"""
import numpy as np
from numba import jit, prange
from Operator import *
from typing import Callable, TypeVar

T = TypeVar("T")


def context_operator_function(e: Operator, i: int, f_k: Callable[[int, float, float], float]) -> float:
    """
    :param e:    Operator                                            -- Contextual Edit Operator
    :param i:    Integer                                             -- Index in S
    :param f_k:  (int k * int x * float sigma  -> Float in [0,1])    -- Context Function

    :return:     Float                                               -- Value of \phi_e(x) in [0,1]
    """
    if (e.op == MOD):
        return f_k(e.k, i, len(e.S))
    elif (e.op == ADD):
        if (i <= e.k - 1):
            return f_k(e.k, i + 1, len(e.S))
        else:
            return f_k(e.k, i, len(e.S))
    else:  # Case e == DEL
        if (i <= e.k - 1):
            return f_k(e.k, i + 1, len(e.S))
        elif (i == e.k):
            return 0
        else:
            return f_k(e.k, i - 1, len(e.S))


def edit_cost(e, f_k: Callable[[int, float, float], float], sim, alpha: float) -> float:
    """
    :param e:       Operator(Edit_operator * List<T> * T * Int)     -- Contextual Edit Operation
    :param sim:     (T * T) -> Float                                -- Similarity measure
    :param f_k:     (Int * Float * Float)  -> Float                 -- Context Function
    :param alpha:   Float

    :return:        Float                                           --  \gamma(e) in [0,1]
    """
    sim_v = np.zeros_like(e.S)  # Vector of similarity between `a` and all symbols in S weighted by context fun
    for i in prange(sim_v.size):
        sim_v[i] = sim(e.S[i], e.a) * context_operator_function(e, i, f_k)

    delta_cost = 1 if e.op != MOD else 1 - sim(e.S[e.k], e.a)  # Fixed cost
    gamma = (alpha * delta_cost) + (1 - alpha) * (1 - sim_v.max())
    return gamma


def one_sided_CED(S1, S2, sim, f_k: Callable[[int, float, float], float], alpha: float):
    """
    :return: Float                                                  -- \tilde{d}_{CED}(S_1,S_2) Cost to edit S1 -> S2
    """
    S1 = np.array(S1, dtype=object)
    S2=np.array(S2, dtype=object)
    D = np.zeros((S1.size + 1, S2.size + 1))
    for i in prange(S1.size + 1):
        for j in prange(S2.size + 1):
            if (i == 0 or j == 0):
                D[i, j] = j + i
            else:
                op_mod = Operator(MOD, S1, S2[j - 1], i - 1)  # Modification operator
                op_del = Operator(DEL, S1, S1[i - 1], i - 1)  # Deletion --
                op_add = Operator(ADD, S1, S2[j - 1], i - 1)  # Addition --

                cost_mod = edit_cost(op_mod, f_k, sim, alpha)  # Cost of apply Modification
                cost_del = edit_cost(op_del, f_k, sim, alpha)  # --            Del
                cost_add = edit_cost(op_add, f_k, sim, alpha)  # --            Add

                D[i, j] = round(min(D[i - 1, j - 1] + cost_mod,
                                    D[i - 1, j] + cost_del,
                                    D[i, j - 1] + cost_add), 2)
    return D[S1.size, S2.size]


def ced(S1, S2, sim, f_k: Callable[[int, float, float], float], alpha: float=0.0):
    """
    :param S1: NumpyArray<T>                                        -- Sequence S1
    :param S2: NumpyArray<T>                                        -- Sequence S2
    :param sim: (T * T) -> Float                                    -- Similarity measure
    :param f_k: (Int * Float * Float)  -> Float                     -- Context Function
    :param alpha: Float

    :return: Float                                                  -- Cost to edit S1 -> S2
    """
    return max(one_sided_CED(S1, S2, sim, f_k, alpha), one_sided_CED(S2, S1, sim, f_k, alpha))


def ced_sm(SM1, SM2, sim, sim_set, f_k: Callable[[int, float, float], float], alpha=0, beta=0.5):
    return beta * ced(SM1[0], SM2[0], sim, f_k, alpha) + (1 - beta) * ced(SM1[1], SM2[1], sim_set, f_k, alpha)
