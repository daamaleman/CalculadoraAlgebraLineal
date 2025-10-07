from __future__ import annotations
from fractions import Fraction
from typing import List, Tuple, Optional

Number = Fraction

def parse_number(token: str) -> Number:
    """Parse user input like '3', '-1.5', '2/3' into exact Fraction."""
    token = token.strip().replace(',', '.')  # allow comma decimals
    if '/' in token:
        parts = token.split('/')
        if len(parts) != 2:
            raise ValueError(f"Fracción inválida: '{token}'. Usa el formato a/b, con a y b números.")
        num, den = parts
        try:
            num_f = Fraction(num)
            den_f = Fraction(den)
            if den_f == 0:
                raise ValueError('El denominador no puede ser cero')
            return num_f / den_f
        except Exception:
            raise ValueError(f"Fracción inválida: '{token}'. Usa el formato a/b, con a y b números.")
    if token == '' or token == '+':
        raise ValueError('Número vacío')
    try:
        return Fraction(token)
    except Exception:
        raise ValueError(f"Número inválido: '{token}'")

class Matrix:
    """Simple immutable Matrix wrapper using Fraction arithmetic."""
    def __init__(self, rows: List[List[Number]]):
        if not rows or not rows[0]:
            raise ValueError("Matrix cannot be empty")
        width = len(rows[0])
        for r in rows:
            if len(r) != width:
                raise ValueError("All rows must have the same length")
        # Deep copy to avoid external mutation
        self._rows: Tuple[Tuple[Number, ...], ...] = tuple(tuple(Fraction(x) for x in r) for r in rows)

    @property
    def m(self) -> int:  # rows
        return len(self._rows)

    @property
    def n(self) -> int:  # cols
        return len(self._rows[0])

    def at(self, i: int, j: int) -> Number:
        return self._rows[i][j]

    def rows(self) -> List[List[Number]]:
        return [list(r) for r in self._rows]

    # Basic operations
    def add(self, other: 'Matrix') -> 'Matrix':
        self._check_same_size(other)
        return Matrix([[self.at(i,j) + other.at(i,j) for j in range(self.n)] for i in range(self.m)])

    def sub(self, other: 'Matrix') -> 'Matrix':
        self._check_same_size(other)
        return Matrix([[self.at(i,j) - other.at(i,j) for j in range(self.n)] for i in range(self.m)])

    def scalar(self, k: Number) -> 'Matrix':
        k = Fraction(k)
        return Matrix([[self.at(i,j) * k for j in range(self.n)] for i in range(self.m)])

    def mul(self, other: 'Matrix') -> 'Matrix':
        if self.n != other.m:
            raise ValueError("Incompatible sizes for multiplication")
        res = []
        for i in range(self.m):
            row = []
            for j in range(other.n):
                s = Fraction(0, 1)
                for k in range(self.n):
                    s += self.at(i, k) * other.at(k, j)
                row.append(s)
            res.append(row)
        return Matrix(res)

    def transpose(self) -> 'Matrix':
        return Matrix([[self.at(i, j) for i in range(self.m)] for j in range(self.n)])

    @staticmethod
    def identity(n: int) -> 'Matrix':
        return Matrix([[Fraction(1 if i == j else 0, 1) for j in range(n)] for i in range(n)])

    def augmented_with(self, b: 'Matrix') -> 'Matrix':
        if self.m != b.m:
            raise ValueError('Augmented matrix must have same number of rows')
        return Matrix([list(self._rows[i]) + list(b._rows[i]) for i in range(self.m)])

    def _check_same_size(self, other: 'Matrix'):
        if self.m != other.m or self.n != other.n:
            raise ValueError("Matrices must have the same dimensions")

    # Pretty helpers
    def to_str(self) -> str:
        return '\n'.join([' '.join([str(x) for x in row]) for row in self._rows])

    def __repr__(self):
        return f"Matrix({self.rows()})"
