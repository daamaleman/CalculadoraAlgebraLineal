from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
from fractions import Fraction
from dataclasses import dataclass
from .matrix import Matrix, Number
from .formatter import block_from_matrix, with_op

@dataclass
class Step:
    matrix: Matrix
    op: Optional[str]

class LinearSystemSolver:
    """Gaussian/Gauss–Jordan elimination with exact Fractions and step capture."""
    def __init__(self, augmented: Matrix):
        self._aug = augmented

    def rref(self) -> Tuple[Matrix, List[Step], int, int]:
        A = [row[:] for row in self._aug.rows()]
        m = len(A); n = len(A[0])
        r = 0
        c = 0
        steps: List[Step] = [Step(Matrix([row[:] for row in A]), op=None)]
        while r < m and c < n:
            # find pivot
            piv = None
            for i in range(r, m):
                if A[i][c] != 0:
                    piv = i; break
            if piv is None:
                c += 1
                continue
            if piv != r:
                A[r], A[piv] = A[piv], A[r]
                steps.append(Step(Matrix([row[:] for row in A]), op=f"f{r+1} ↔ f{piv+1}"))
            # normalize pivot row
            pivval = A[r][c]
            if pivval != 1:
                A[r] = [x / pivval for x in A[r]]
                steps.append(Step(Matrix([row[:] for row in A]), op=f"f{r+1} ← (1/{pivval}) f{r+1}"))
            # eliminate other rows
            for i in range(m):
                if i == r:
                    continue
                if A[i][c] != 0:
                    k = A[i][c]
                    A[i] = [A[i][j] - k * A[r][j] for j in range(n)]
                    ks = f"{k}" if k.denominator == 1 else f"{k.numerator}/{k.denominator}"
                    steps.append(Step(Matrix([row[:] for row in A]), op=f"f{i+1} ← f{i+1} - ({ks}) f{r+1}"))
            r += 1; c += 1
        # rank estimate (not heavily used later)
        rank = 0
        for i in range(m):
            if any(self._aug.at(i, j) != 0 for j in range(n)):
                rank += 1
        return Matrix([row[:] for row in A]), steps, rank, m

    def solve(self, n_vars: int) -> Dict[str, Any]:
        rref_mat, steps, rank, m = self.rref()
        rows = rref_mat.rows()
        # inconsistency: [0 ... 0 | b] with b != 0
        inconsistent = False
        for row in rows:
            if all(x == 0 for x in row[:n_vars]) and row[n_vars] != 0:
                inconsistent = True; break
        if inconsistent:
            return {"type": "inconsistent", "steps": steps, "rref": rref_mat, "pivot_cols": [], "free_vars": list(range(n_vars))}
        # count pivots
        pivots = 0
        pivot_cols = []
        for j in range(n_vars):
            col = [rows[i][j] for i in range(m)]
            if any(col) and any(rows[i][j] == 1 and all(rows[i][k] == 0 for k in range(n_vars) if k != j) for i in range(m)):
                pivots += 1; pivot_cols.append(j)
        free_vars = [j for j in range(n_vars) if j not in pivot_cols]
        if pivots == n_vars:
            # unique solution: read last column (assuming augmented [A | b])
            sol = [rows[i][-1] for i in range(n_vars)]
            return {"type": "unique", "solution": sol, "steps": steps, "rref": rref_mat, "pivot_cols": pivot_cols, "free_vars": free_vars}
        else:
            # infinite solutions (basic notification). Parametrization puede ser agregada.
            return {"type": "infinite", "steps": steps, "rref": rref_mat, "pivot_cols": pivot_cols, "free_vars": free_vars}
