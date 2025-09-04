from __future__ import annotations
from typing import List, Optional
from fractions import Fraction
from .matrix import Matrix

def frac_to_str(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"

def pretty_matrix(mat: Matrix) -> List[str]:
    """Return list of strings representing a bracketed matrix with padded columns."""
    rows = mat.rows()
    cols = mat.n
    colw = [0] * cols
    cells = [[frac_to_str(x) for x in r] for r in rows]
    for j in range(cols):
        colw[j] = max(len(cells[i][j]) for i in range(mat.m))
    lines = []
    for i in range(mat.m):
        content = ' '.join(cells[i][j].rjust(colw[j]) for j in range(cols))
        lines.append(f"| {content} |")
    # Big brackets
    if len(lines) == 1:
        return ['⎡' + lines[0][1:-1] + '⎤']
    return (
        ['⎡' + lines[0][1:-1] + '⎤'] +
        ['⎢' + line[1:-1] + '⎥' for line in lines[1:-1]] +
        ['⎣' + lines[-1][1:-1] + '⎦']
    )

def join_augmented(left: Matrix, right: Matrix) -> List[str]:
    """Side-by-side matrices with a vertical bar separator to mimic augmented matrix."""
    L = pretty_matrix(left)
    R = pretty_matrix(right)
    h = max(len(L), len(R))
    L += [' ' * len(L[0])] * (h - len(L))
    R += [' ' * len(R[0])] * (h - len(R))
    bar = ['  │  '] * h
    return [L[i] + bar[i] + R[i] for i in range(h)]

def block_from_matrix(mat: Matrix) -> List[str]:
    return pretty_matrix(mat)

def with_op(lines: List[str], op: Optional[str]) -> str:
    """Attach operation text to the right side of the block."""
    if not lines:
        return ''
    pad = max(len(line) for line in lines) + 2
    if op is None:
        return '\n'.join(lines)
    mid = len(lines) // 2
    out = []
    for i, line in enumerate(lines):
        if i == mid:
            out.append(line.ljust(pad) + f"⟶  {op}")
        else:
            out.append(line)
    return '\n'.join(out)
