# processor_type/interface.py

"""
Абстрактный интерфейс backend.
Определяет контракт: какие операции должен поддерживать любой backend.
"""

from abc import ABC, abstractmethod

from core.dense_tensor import DenseTensor


Number = int | float


class BackendInterface(ABC):
    """
    Абстрактный backend для вычислений.
    Любой конкретный backend (CPU, GPU) должен реализовать все эти методы.
    """

    # ────────────────────────────────────────────
    # Создание тензоров
    # ────────────────────────────────────────────

    @abstractmethod
    def zeros(self, shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """Создаёт тензор, заполненный нулями."""
        pass

    @abstractmethod
    def ones(self, shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """Создаёт тензор, заполненный единицами."""
        pass

    @abstractmethod
    def eye(self, n: int) -> DenseTensor:
        """Создаёт единичную матрицу n x n."""
        pass

    @abstractmethod
    def random(
        self,
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """Создаёт тензор со случайными значениями."""
        pass

    @abstractmethod
    def from_dense_tensor(self, dense_tensor: DenseTensor) -> DenseTensor:
        """Оборачивает DenseTensor в формат, совместимый с backend."""
        pass

    @abstractmethod
    def copy(self, tensor: DenseTensor) -> DenseTensor:
        """Глубокая копия тензора."""
        pass

    # ────────────────────────────────────────────
    # Информация о тензоре
    # ────────────────────────────────────────────

    @abstractmethod
    def shape(self, tensor: DenseTensor) -> tuple[int, ...]:
        """Возвращает shape тензора."""
        pass

    @abstractmethod
    def size(self, tensor: DenseTensor) -> int:
        """Возвращает общее число элементов."""
        pass

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    @abstractmethod
    def reshape(
        self,
        tensor: DenseTensor,
        new_shape: tuple[int, ...] | list[int]
    ) -> DenseTensor:
        """Изменение формы тензора (новый тензор с копией данных)."""
        pass

    @abstractmethod
    def transpose(self, matrix: DenseTensor) -> DenseTensor:
        """Транспонирование 2D-матрицы."""
        pass

    # ────────────────────────────────────────────
    # Поэлементная арифметика
    # ────────────────────────────────────────────

    @abstractmethod
    def add(self, a: DenseTensor, b: DenseTensor) -> DenseTensor:
        """Поэлементное сложение."""
        pass

    @abstractmethod
    def sub(self, a: DenseTensor, b: DenseTensor) -> DenseTensor:
        """Поэлементное вычитание."""
        pass

    @abstractmethod
    def scale(self, tensor: DenseTensor, alpha: Number) -> DenseTensor:
        """Умножение тензора на скаляр."""
        pass

    # ────────────────────────────────────────────
    # Линейная алгебра
    # ────────────────────────────────────────────

    @abstractmethod
    def matmul(self, a: DenseTensor, b: DenseTensor) -> DenseTensor:
        """
        Матричное произведение строка на столбец: C = A @ B.
        """
        pass

    @abstractmethod
    def svd(
        self,
        matrix: DenseTensor,
        full_matrices: bool = False
    ) -> tuple[DenseTensor, DenseTensor, DenseTensor]:
        """
        Тонкое SVD-разложение матрицы.
        matrix: (m, n)

        Возвращает:
            U:  (m, k) — левые сингулярные векторы
            S:  (k,)   — сингулярные значения (по убыванию)
            Vt: (k, n) — правые сингулярные векторы (транспонированные)

        где k = min(m, n).
        """
        pass

    @abstractmethod
    def qr(self, matrix: DenseTensor) -> tuple[DenseTensor, DenseTensor]:
        """
        Тонкое QR-разложение матрицы.
        matrix: (m, n), m >= n

        Возвращает:
            Q: (m, n) — ортогональные столбцы
            R: (n, n) — верхнетреугольная
        """
        pass

    @abstractmethod
    def norm(self, tensor: DenseTensor) -> float:
        """Фробениусова норма тензора."""
        pass

    @abstractmethod
    def diag(self, vector: DenseTensor) -> DenseTensor:
        """
        Создаёт диагональную матрицу из вектора.
        vector: (n,) -> matrix: (n, n)
        """
        pass

    @abstractmethod
    def get_element(
        self,
        tensor: DenseTensor,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """Доступ к одному элементу тензора."""
        pass

    @abstractmethod
    def set_element(
        self,
        tensor: DenseTensor,
        indices: tuple[int, ...] | list[int],
        value: Number
    ) -> DenseTensor:
        """Установка одного элемента тензора."""
        pass