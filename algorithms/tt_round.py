# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    if max_rank is not None and max_rank <= 0:
        raise ValueError("max_rank должен быть положительным числом или None")

    if eps < 0:
        raise ValueError("eps должен быть неотрицательным")

    rounded = right_canonicalize(tt, backend)
    cores = [core.copy() for core in rounded.cores]

    if rounded.order == 1:
        return TTTensor(cores)

    first_norm = cores[0].norm()

    if first_norm > 1e-30:
        delta = eps * first_norm / math.sqrt(rounded.order - 1)
    else:
        delta = 0.0

    for k in range(rounded.order - 1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        matrix = backend.reshape(core, (r_left * n_k, r_right))
        U, S, Vt = backend.svd(matrix, full_matrices=False)

        rank = _compute_rank(S, delta, max_rank)

        U_trunc = _truncate_columns(U, rank, backend)
        S_trunc = _truncate_vector(S, rank, backend)
        Vt_trunc = _truncate_rows(Vt, rank, backend)

        cores[k] = backend.reshape(U_trunc, (r_left, n_k, rank))

        rest = _multiply_diag_matrix(S_trunc, Vt_trunc, rank, backend)

        next_core = cores[k + 1]
        _, n_next, r_next = next_core.shape
        new_next_core = backend.zeros((rank, n_next, r_next))

        for i in range(n_next):
            slice_data = []
            for row in range(r_right):
                for col in range(r_next):
                    slice_data.append(next_core[row, i, col])

            slice_matrix = DenseTensor((r_right, r_next), data=slice_data)
            product = backend.matmul(rest, slice_matrix)

            for row in range(rank):
                for col in range(r_next):
                    new_next_core[row, i, col] = product[row, col]

        cores[k + 1] = new_next_core

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть одномерным тензором")

    if S.size == 0:
        return 1

    numerical_rank = 0
    max_s = abs(S[0])
    threshold = max(1e-12, 1e-8 * max_s)

    for i in range(S.size):
        if abs(S[i]) > threshold:
            numerical_rank += 1

    numerical_rank = max(1, numerical_rank)

    rank = numerical_rank

    if delta > 0:
        tail_sum = 0.0
        rank = numerical_rank

        while rank > 1:
            tail_sum += S[rank - 1] * S[rank - 1]

            if tail_sum <= delta * delta:
                rank -= 1
            else:
                break

    if max_rank is not None:
        rank = min(rank, max_rank)

    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным тензором")

    m, n = matrix.shape

    if rank < 0 or rank > n:
        raise ValueError("rank выходит за границы числа столбцов")

    result = backend.zeros((m, rank))

    for i in range(m):
        for j in range(rank):
            result[i, j] = matrix[i, j]

    return result


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным тензором")

    m, n = matrix.shape

    if rank < 0 or rank > m:
        raise ValueError("rank выходит за границы числа строк")

    result = backend.zeros((rank, n))

    for i in range(rank):
        for j in range(n):
            result[i, j] = matrix[i, j]

    return result


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    if vector.ndim != 1:
        raise ValueError("vector должен быть одномерным тензором")

    if rank < 0 or rank > vector.size:
        raise ValueError("rank выходит за границы длины вектора")

    result = backend.zeros((rank,))

    for i in range(rank):
        result[i] = vector[i]

    return result


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть одномерным тензором")

    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным тензором")

    if rank < 0 or rank > diag_vec.size:
        raise ValueError("rank выходит за границы diag_vec")

    if matrix.shape[0] != rank:
        raise ValueError("число строк matrix должно быть равно rank")

    n = matrix.shape[1]
    result = backend.zeros((rank, n))

    for i in range(rank):
        for j in range(n):
            result[i, j] = diag_vec[i] * matrix[i, j]

    return result