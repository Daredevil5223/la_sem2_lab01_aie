# processor_type/cpu_backend.py

"""
CPU backend: фасад для выполнения операций на CPU.
"""

from processor_type.interface import BackendInterface
from core.dense_tensor import DenseTensor
from core import linalg


class CPUBackend(BackendInterface):

    # ────────────────────────────────────────────
    # Создание тензоров
    # ────────────────────────────────────────────

    def zeros(self, shape: tuple[int, ...]) -> DenseTensor:
        return DenseTensor.zeros(shape)

    def ones(self, shape: tuple[int, ...]) -> DenseTensor:
        return DenseTensor.ones(shape)

    def eye(self, n: int) -> DenseTensor:
        return linalg.eye(n)

    def random(
        self,
        shape: tuple[int, ...],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        return DenseTensor.random(
            shape,
            low=low,
            high=high,
            integer=integer,
            seed=seed,
        )

    def from_dense_tensor(self, dense_tensor: DenseTensor) -> DenseTensor:
        return dense_tensor

    def copy(self, tensor: DenseTensor) -> DenseTensor:
        return tensor.copy()

    # ────────────────────────────────────────────
    # Информация о тензоре
    # ────────────────────────────────────────────

    def shape(self, tensor: DenseTensor) -> tuple[int, ...]:
        return tensor.shape

    def size(self, tensor: DenseTensor) -> int:
        return tensor.size

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(
        self,
        tensor: DenseTensor,
        new_shape: tuple[int, ...]
    ) -> DenseTensor:
        return tensor.reshape(new_shape)

    def transpose(self, matrix: DenseTensor) -> DenseTensor:
        return linalg.transpose(matrix)

    # ────────────────────────────────────────────
    # Поэлементная арифметика
    # ────────────────────────────────────────────

    def add(self, a: DenseTensor, b: DenseTensor) -> DenseTensor:
        return a + b

    def sub(self, a: DenseTensor, b: DenseTensor) -> DenseTensor:
        return a - b

    def scale(
        self,
        tensor: DenseTensor,
        alpha: int | float
    ) -> DenseTensor:
        return tensor * alpha

    # ────────────────────────────────────────────
    # Линейная алгебра
    # ────────────────────────────────────────────

    def matmul(self, a: DenseTensor, b: DenseTensor) -> DenseTensor:
        return linalg.matmul(a, b)

    def svd(
        self,
        matrix: DenseTensor,
        full_matrices: bool = False
    ) -> tuple[DenseTensor, DenseTensor, DenseTensor]:
        return linalg.svd(matrix, full_matrices=full_matrices)

    def qr(self, matrix: DenseTensor) -> tuple[DenseTensor, DenseTensor]:
        return linalg.qr(matrix)

    def norm(self, tensor: DenseTensor) -> float:
        return tensor.norm()

    def diag(self, vector: DenseTensor) -> DenseTensor:
        return linalg.diag(vector)

    def get_element(
        self,
        tensor: DenseTensor,
        indices: tuple[int, ...]
    ) -> float:
        return tensor[tuple(indices)]

    def set_element(
        self,
        tensor: DenseTensor,
        indices: tuple[int, ...],
        value: float
    ) -> DenseTensor:
        tensor[tuple(indices)] = value
        return tensor