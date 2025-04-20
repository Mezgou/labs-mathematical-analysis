import numpy as np
from abc import ABC, abstractmethod


class Integrator(ABC):
    """
    Абстрактный базовый класс для численных методов интегрирования.
    Все подклассы должны реализовывать метод integrate.
    """
    @abstractmethod
    def integrate(self, a: float, b: float, n: int, f: callable) -> float:
        """
        Вычисляет приближённое значение определённого интеграла функции f
        на отрезке [a, b] с использованием n разбиений.
        """
        pass


class RectangleMethod(Integrator):
    """
    Метод прямоугольников для численного интегрирования.
    
    Поддерживаются 4 режима:
    - "left": значение функции в левой точке подынтервала
    - "right": значение функции в правой точке подынтервала
    - "mid": значение функции в средней точке подынтервала
    - "random": случайная точка внутри подынтервала
    """

    def __init__(self, mode: str = "left") -> None:
        assert mode in ('left', 'right', 'mid', 'random')
        self.mode = mode

    def integrate(self, a: float, b: float, n: int, f: callable) -> float:
        """
        Вычисляет интеграл методом прямоугольников согласно выбранному режиму.

        Параметры:
        - a, b: границы интегрирования
        - n: количество разбиений
        - f: интегрируемая функция

        Возвращает:
        - Приближённое значение интеграла
        """
        dx: float = (b - a) / n
        total: float = 0.0
        for i in range(n):
            x0: float = a + i * dx
            x1: float = x0 + dx
            if self.mode == "left":
                xi: float = x0
            elif self.mode == "right":
                xi: float = x1
            elif self.mode == "mid":
                xi: float = (x0 + x1) / 2
            else:
                xi: float = np.random.uniform(x0, x1)
            total += f(xi) * dx
        return total


class TrapezoidalMethod(Integrator):
    """
    Метод трапеций для численного интегрирования.

    Использует среднее значение функции на концах каждого подынтервала.
    """

    def integrate(self, a: float, b: float, n: int, f: callable) -> float:
        """
        Вычисляет интеграл методом трапеций.

        Параметры:
        - a, b: границы интегрирования
        - n: количество разбиений
        - f: интегрируемая функция

        Возвращает:
        - Приближённое значение интеграла
        """
        dx: float = (b - a) / n
        total: float = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            total += f(a + i * dx)
        return total * dx


class SimpsonMethod(Integrator):
    """
    Метод Симпсона (параболическая аппроксимация) для численного интегрирования.

    Требует чётное число разбиений.
    """

    def integrate(self, a: float, b: float, n: int, f: callable) -> float:
        """
        Вычисляет интеграл методом Симпсона.

        Параметры:
        - a, b: границы интегрирования
        - n: количество разбиений (должно быть чётным)
        - f: интегрируемая функция

        Возвращает:
        - Приближённое значение интеграла
        """
        if n % 2 != 0:
            raise ValueError("For the Simpson method, n must be even")
        dx: float = (b - a) / n
        total: float = f(a) + f(b)
        for i in range(1, n):
            coef: float = 4 if i % 2 != 0 else 2
            total += coef * f(a + i * dx)
        return total * dx / 3
