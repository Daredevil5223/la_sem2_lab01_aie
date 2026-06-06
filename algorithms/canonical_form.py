# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order - 1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        matrix = backend.reshape(core, (r_left * n_k, r_right))
        Q, R = backend.qr(matrix)

        new_rank = Q.shape[1]
        cores[k] = backend.reshape(Q, (r_left, n_k, new_rank))

        next_core = cores[k + 1]
        _, n_next, r_next = next_core.shape
        new_next_core = DenseTensor.zeros((new_rank, n_next, r_next))

        for i in range(n_next):
            slice_data = []
            for row in range(r_right):
                for col in range(r_next):
                    slice_data.append(next_core[row, i, col])

            slice_matrix = DenseTensor((r_right, r_next), data=slice_data)
            product = backend.matmul(R, slice_matrix)

            for row in range(new_rank):
                for col in range(r_next):
                    new_next_core[row, i, col] = product[row, col]

        cores[k + 1] = new_next_core

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]

    for k in range(tt.order - 1, 0, -1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        matrix = backend.reshape(core, (r_left, n_k * r_right))

        if n_k * r_right >= r_left:
            transposed = backend.transpose(matrix)
            Q, R = backend.qr(transposed)

            Q_t = backend.transpose(Q)
            R_t = backend.transpose(R)

            new_rank = Q_t.shape[0]
            cores[k] = backend.reshape(Q_t, (new_rank, n_k, r_right))

            prev_core = cores[k - 1]
            r_prev, n_prev, _ = prev_core.shape
            new_prev_core = DenseTensor.zeros((r_prev, n_prev, new_rank))

            for i in range(n_prev):
                slice_data = []
                for row in range(r_prev):
                    for col in range(r_left):
                        slice_data.append(prev_core[row, i, col])

                slice_matrix = DenseTensor((r_prev, r_left), data=slice_data)
                product = backend.matmul(slice_matrix, R_t)

                for row in range(r_prev):
                    for col in range(new_rank):
                        new_prev_core[row, i, col] = product[row, col]

            cores[k - 1] = new_prev_core

        else:
            U, S, Vt = backend.svd(matrix, full_matrices=False)

            rank = _numerical_rank(S)
            rank = max(1, rank)

            U_trunc = _truncate_columns(U, rank, backend)
            S_trunc = _truncate_vector(S, rank, backend)
            Vt_trunc = _truncate_rows(Vt, rank, backend)

            cores[k] = backend.reshape(Vt_trunc, (rank, n_k, r_right))

            left_factor = _multiply_columns_by_diag(U_trunc, S_trunc, backend)

            prev_core = cores[k - 1]
            r_prev, n_prev, _ = prev_core.shape
            new_prev_core = DenseTensor.zeros((r_prev, n_prev, rank))

            for i in range(n_prev):
                slice_data = []
                for row in range(r_prev):
                    for col in range(r_left):
                        slice_data.append(prev_core[row, i, col])

                slice_matrix = DenseTensor((r_prev, r_left), data=slice_data)
                product = backend.matmul(slice_matrix, left_factor)

                for row in range(r_prev):
                    for col in range(rank):
                        new_prev_core[row, i, col] = product[row, col]

            cores[k - 1] = new_prev_core

    return TTTensor(cores)





# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    if S.ndim != 1:
        raise ValueError("S должен быть одномерным тензором")

    if S.size == 0:
        return 0

    max_s = max(abs(x) for x in S.data)
    threshold = max(abs_tol, rel_tol * max_s)

    rank = 0
    for value in S.data:
        if abs(value) > threshold:
            rank += 1

    return rank


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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
        rank:     длина диагонального вектора
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
        scale = diag_vec[i]
        for j in range(n):
            result[i, j] = scale * matrix[i, j]

    return result


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    if matrix.ndim != 2:
        raise ValueError("matrix должен быть двумерным тензором")

    if diag_vec.ndim != 1:
        raise ValueError("diag_vec должен быть одномерным тензором")

    m, n = matrix.shape

    if diag_vec.size != n:
        raise ValueError("длина diag_vec должна совпадать с числом столбцов matrix")

    result = backend.zeros((m, n))

    for i in range(m):
        for j in range(n):
            result[i, j] = matrix[i, j] * diag_vec[j]

    return result