# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковый shape")

    cores = []
    order = tt1.order

    for k in range(order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n_k, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        if k == 0:
            new_core = backend.zeros((1, n_k, r1_right + r2_right))

            for i in range(n_k):
                for right in range(r1_right):
                    new_core[0, i, right] = core1[0, i, right]
                for right in range(r2_right):
                    new_core[0, i, r1_right + right] = core2[0, i, right]

        elif k == order - 1:
            new_core = backend.zeros((r1_left + r2_left, n_k, 1))

            for i in range(n_k):
                for left in range(r1_left):
                    new_core[left, i, 0] = core1[left, i, 0]
                for left in range(r2_left):
                    new_core[r1_left + left, i, 0] = core2[left, i, 0]

        else:
            new_core = backend.zeros(
                (r1_left + r2_left, n_k, r1_right + r2_right)
            )

            for i in range(n_k):
                for left in range(r1_left):
                    for right in range(r1_right):
                        new_core[left, i, right] = core1[left, i, right]

                for left in range(r2_left):
                    for right in range(r2_right):
                        new_core[
                            r1_left + left,
                            i,
                            r1_right + right
                        ] = core2[left, i, right]

        cores.append(new_core)

    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """

    if not isinstance(alpha, (int, float)):
        raise TypeError("alpha должен быть числом")

    cores = [core.copy() for core in tt.cores]
    cores[0] = backend.scale(cores[0], alpha)

    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковый shape")

    cores = []

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n_k, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        new_core = backend.zeros(
            (r1_left * r2_left, n_k, r1_right * r2_right)
        )
        for i in range(n_k):
            for left1 in range(r1_left):
                for left2 in range(r2_left):
                    new_left = left1 * r2_left + left2

                    for right1 in range(r1_right):
                        for right2 in range(r2_right):
                            new_right = right1 * r2_right + right2
                            new_core[new_left, i, new_right] = (
                                    core1[left1, i, right1]
                                    * core2[left2, i, right2]
                            )

        cores.append(new_core)

    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковый shape")

    z = backend.ones((1, 1))

    for k in range(tt1.order):
        core1 = tt1.cores[k]
        core2 = tt2.cores[k]

        r1_left, n_k, r1_right = core1.shape
        r2_left, _, r2_right = core2.shape

        new_z = backend.zeros((r1_right, r2_right))

        for right1 in range(r1_right):
            for right2 in range(r2_right):
                value = 0.0

                for left1 in range(r1_left):
                    for left2 in range(r2_left):
                        z_value = z[left1, left2]

                        for i in range(n_k):
                            value += (
                                    core1[left1, i, right1]
                                    * z_value
                                    * core2[left2, i, right2]
                            )

                new_z[right1, right2] = value

        z = new_z

    return z[0, 0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    value = tt_dot(tt, tt, backend)

    if value < 0 and abs(value) < 1e-12:
        value = 0.0

    return math.sqrt(value)


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    if tt1.shape != tt2.shape:
        raise ValueError("TT-тензоры должны иметь одинаковый shape")

    value = (
            tt_dot(tt1, tt1, backend)
            - 2.0 * tt_dot(tt1, tt2, backend)
            + tt_dot(tt2, tt2, backend)
    )

    if value < 0 and abs(value) < 1e-12:
        value = 0.0

    return math.sqrt(value)