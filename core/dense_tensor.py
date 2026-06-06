# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
    check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        self.shape = validate_shape(shape)
        self.ndim = len(self.shape)
        self.size = compute_size(self.shape)
        self.strides = compute_strides(self.shape)

        if data is None:
            self.data = [float(fill)] * self.size
        else:
            if len(data) != self.size:
                raise ValueError(
                    f"Размер data не совпадает с shape: {len(data)} != {self.size}"
                )
            self.data = [float(x) for x in data]


    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        shape = validate_shape(shape)
        size = compute_size(shape)

        rng = random.Random(seed)

        if integer:
            data = [float(rng.randint(low, high)) for _ in range(size)]
        else:
            data = [rng.uniform(low, high) for _ in range(size)]

        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """

        def get_shape(obj: list) -> tuple[int, ...]:
            if not isinstance(obj, list):
                return ()

            if len(obj) == 0:
                raise ValueError("Вложенные списки не должны быть пустыми")

            first_shape = get_shape(obj[0])

            for item in obj:
                if get_shape(item) != first_shape:
                    raise ValueError("Вложенный список должен быть прямоугольным")

            return (len(obj),) + first_shape

        def flatten(obj: list) -> list[float]:
            if not isinstance(obj, list):
                return [float(obj)]

            result = []
            for item in obj:
                result.extend(flatten(item))
            return result

        shape = get_shape(nested)
        data = flatten(nested)

        return DenseTensor(shape, data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            multi_index = (multi_index,)
        elif not isinstance(multi_index, tuple):
            raise TypeError("Индекс должен быть int или tuple")

        if len(multi_index) != self.ndim:
            raise IndexError(
                f"Ожидалось {self.ndim} индексов, получено {len(multi_index)}"
            )

        for index, dim in zip(multi_index, self.shape):
            if not isinstance(index, int):
                raise TypeError("Все индексы должны быть целыми числами")
            if index < 0 or index >= dim:
                raise IndexError("Индекс выходит за границы тензора")

        return multi_index

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        multi_index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(multi_index, self.strides)
        return self.data[flat_index]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        multi_index = self._validate_index(multi_index)
        flat_index = multi_index_to_flat(multi_index, self.strides)
        self.data[flat_index] = float(value)

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape = validate_shape(new_shape)

        if compute_size(new_shape) != self.size:
            raise ValueError(
                f"Нельзя reshape из {self.shape} в {new_shape}: размеры не совпадают"
            )

        return DenseTensor(new_shape, data=self.data[:])

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if not isinstance(mode, int):
            raise TypeError("mode должен быть целым числом")

        if mode < 0 or mode >= self.ndim:
            raise ValueError("mode выходит за границы тензора")

        rows = self.shape[mode]
        cols = self.size // rows
        result = DenseTensor.zeros((rows, cols))

        other_shape = self.shape[:mode] + self.shape[mode + 1:]
        other_strides = compute_strides(other_shape) if len(other_shape) > 0 else ()

        for flat_index, value in enumerate(self.data):
            multi_index = flat_to_multi_index(flat_index, self.shape)
            row = multi_index[mode]
            other_index = multi_index[:mode] + multi_index[mode + 1:]

            if len(other_index) == 0:
                col = 0
            else:
                col = multi_index_to_flat(other_index, other_strides)

            result[row, col] = value

        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if not isinstance(k, int):
            raise TypeError("k должен быть целым числом")

        if k < 0 or k >= self.ndim - 1:
            raise ValueError("k выходит за допустимые границы")

        left_size = compute_size(self.shape[:k + 1])
        right_size = compute_size(self.shape[k + 1:])

        return DenseTensor((left_size, right_size), data=self.data[:])

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data[:])

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        if not isinstance(other, DenseTensor):
            return NotImplemented

        check_shapes_match(self.shape, other.shape)

        data = [a + b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        if not isinstance(other, DenseTensor):
            return NotImplemented

        check_shapes_match(self.shape, other.shape)

        data = [a - b for a, b in zip(self.data, other.data)]
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        if not isinstance(scalar, (int, float)):
            return NotImplemented

        data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self * -1.0

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if not isinstance(other, DenseTensor):
            return False

        if self.shape != other.shape:
            return False

        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False

        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""

        def build(prefix: tuple[int, ...], axis: int) -> list | float:
            if axis == self.ndim:
                return self[prefix]

            return [
                build(prefix + (i,), axis + 1)
                for i in range(self.shape[axis])
            ]

        return build((), 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.to_nested_list()})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()