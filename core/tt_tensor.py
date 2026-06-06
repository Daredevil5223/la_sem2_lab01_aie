# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        if not isinstance(cores, list):
            raise TypeError("cores должен быть списком")

        if len(cores) == 0:
            raise ValueError("список cores не должен быть пустым")

        for core in cores:
            if not isinstance(core, DenseTensor):
                raise TypeError("каждое ядро должно быть DenseTensor")
            if core.ndim != 3:
                raise ValueError("каждое TT-ядро должно быть 3D-тензором")

        if cores[0].shape[0] != 1:
            raise ValueError("первый TT-ранг должен быть равен 1")

        if cores[-1].shape[2] != 1:
            raise ValueError("последний TT-ранг должен быть равен 1")

        for k in range(len(cores) - 1):
            if cores[k].shape[2] != cores[k + 1].shape[0]:
                raise ValueError("соседние TT-ранги должны совпадать")

        self.cores = cores
        self.order = len(cores)
        self.shape = tuple(core.shape[1] for core in cores)
        self.ranks = tuple([cores[0].shape[0]] + [core.shape[2] for core in cores])

    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shape = validate_shape(shape)
        ranks = tuple(ranks)

        if len(ranks) == len(shape) - 1:
            ranks = (1,) + ranks + (1,)
        elif len(ranks) != len(shape) + 1:
            raise ValueError("некорректная длина ranks")

        if ranks[0] != 1 or ranks[-1] != 1:
            raise ValueError("граничные TT-ранги должны быть равны 1")

        for rank in ranks:
            if not isinstance(rank, int) or rank <= 0:
                raise ValueError("TT-ранги должны быть положительными целыми числами")

        cores = []
        for k in range(len(shape)):
            core_shape = (ranks[k], shape[k], ranks[k + 1])
            core_seed = None if seed is None else seed + k
            cores.append(DenseTensor.random(core_shape, seed=core_seed))

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        if not isinstance(indices, (tuple, list)):
            raise TypeError("indices должен быть tuple или list")

        indices = tuple(indices)

        if len(indices) != self.order:
            raise IndexError("длина indices должна совпадать с порядком тензора")

        for index, dim in zip(indices, self.shape):
            if not isinstance(index, int):
                raise TypeError("все индексы должны быть целыми числами")
            if index < 0 or index >= dim:
                raise IndexError("индекс выходит за границы тензора")

        vector = [1.0]
        for k in range(self.order):
            core = self.cores[k]
            index = indices[k]
            next_rank = self.ranks[k + 1]
            new_vector = [0.0] * next_rank

            for right in range(next_rank):
                value = 0.0
                for left in range(self.ranks[k]):
                    value += vector[left] * core[left, index, right]
                new_vector[right] = value

            vector = new_vector

        return vector[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.zeros(self.shape)

        for flat_index in range(result.size):
            indices = []
            current = flat_index

            for dim in reversed(self.shape):
                indices.append(current % dim)
                current //= dim

            indices.reverse()
            result.data[flat_index] = self.get_element(tuple(indices))

        return result

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        return compute_size(self.shape) / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        return (
            f"TTTensor(\n"
            f"  order={self.order},\n"
            f"  shape={self.shape},\n"
            f"  ranks={self.ranks},\n"
            f"  cores={self.core_sizes()},\n"
            f"  storage={self.total_storage()}\n"
            f")"
        )

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()