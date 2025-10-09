from fractions import Fraction
from typing import List, Tuple, Dict, Any

# Helpers

def _to_fraction(x):
    if isinstance(x, Fraction):
        return x
    try:
        return Fraction(x)
    except Exception:
        return Fraction(str(x))


def clone_matrix(M: List[List[Fraction]]) -> List[List[Fraction]]:
    return [row[:] for row in M]


def rref_augmented(A: List[List[Fraction]], B: List[List[Fraction]]) -> Tuple[List[List[Fraction]], List[int], List[str]]:
    """
    Compute RREF of augmented matrix [A|B].
    Returns (RREF, pivot_cols, steps)
    """
    m = len(A)
    n = len(A[0]) if m else 0
    p = len(B[0]) if B and isinstance(B[0], list) else 1
    # Normalize inputs to Fractions
    A = [[_to_fraction(x) for x in row] for row in A]
    if p == 1 and B and not isinstance(B[0], list):
        B = [[_to_fraction(b)] for b in B]
    else:
        B = [[_to_fraction(x) for x in row] for row in B]

    # Build augmented matrix
    M = [A[i][:] + B[i][:] for i in range(m)]
    steps: List[str] = []

    def fmt_frac(q: Fraction) -> str:
        if q.denominator == 1:
            return str(q.numerator)
        return f"{q.numerator}/{q.denominator}"

    def fmt_matrix(mat: List[List[Fraction]]) -> str:
        # Simple pretty print
        return "\n".join("[ " + ", ".join(fmt_frac(x) for x in row) + " ]" for row in mat)

    steps.append("Matriz aumentada inicial:\n" + fmt_matrix(M))

    r = 0
    c = 0
    pivot_cols: List[int] = []
    total_cols = n + p
    while r < m and c < n:
        # Find pivot row
        pivot = None
        for i in range(r, m):
            if M[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            c += 1
            continue
        if pivot != r:
            M[r], M[pivot] = M[pivot], M[r]
            steps.append(f"Intercambiar R{r+1} ↔ R{pivot+1}\n" + fmt_matrix(M))
        # Scale row r to make pivot 1
        piv = M[r][c]
        if piv != 1:
            M[r] = [x / piv for x in M[r]]
            steps.append(f"R{r+1} ← (1/{fmt_frac(piv)})·R{r+1}\n" + fmt_matrix(M))
        # Eliminate other rows
        for i in range(m):
            if i == r:
                continue
            f = M[i][c]
            if f != 0:
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
                steps.append(f"R{i+1} ← R{i+1} - ({fmt_frac(f)})·R{r+1}\n" + fmt_matrix(M))
        pivot_cols.append(c)
        r += 1
        c += 1

    return M, pivot_cols, steps


def solve_linear_system(A: List[List[Fraction]], B: List[Fraction]) -> Dict[str, Any]:
    """
    Solve A x = B (B is a vector) using RREF, return dict with details.
    Works with Fractions and returns parametric solution when infinite.
    """
    m = len(A)
    n = len(A[0]) if m else 0
    # Normalize B as column matrix
    Bb = [[_to_fraction(b)] for b in B]

    R, pivots, steps = rref_augmented(A, Bb)

    # Consistency check
    inconsistent = False
    for i in range(m):
        if all(R[i][j] == 0 for j in range(n)) and R[i][n] != 0:
            inconsistent = True
            break

    result: Dict[str, Any] = {
        'steps': steps,
        'rref': R,
        'pivots': pivots,
        'consistent': not inconsistent,
        'unique': False,
        'infinite': False,
        'solution': None,
        'param_basis': [],
        'free_vars': [],
        'message': ''
    }

    if inconsistent:
        result['message'] = 'Sistema inconsistente: no hay solución.'
        return result

    rank = len(pivots)
    if rank == n:
        # Unique solution, read from RREF
        x = [Fraction(0) for _ in range(n)]
        for r, c in enumerate(pivots):
            x[c] = R[r][n]
        result['unique'] = True
        result['solution'] = x
        result['message'] = 'Sistema consistente con solución única.'
        return result

    # Infinite solutions: build parametric form
    free = [j for j in range(n) if j not in pivots]
    result['free_vars'] = free
    x_part = [Fraction(0) for _ in range(n)]
    for r, c in enumerate(pivots):
        x_part[c] = R[r][n]
    basis = []
    for f in free:
        v = [Fraction(0) for _ in range(n)]
        v[f] = Fraction(1)
        for r, c in enumerate(pivots):
            v[c] = -R[r][f]
        basis.append(v)
    result['solution'] = x_part
    result['param_basis'] = basis
    result['infinite'] = True
    result['message'] = 'Sistema consistente con infinitas soluciones.'
    return result


def solve_homogeneous(A: List[List[Fraction]]) -> Dict[str, Any]:
    """Solve A x = 0, report if only trivial or infinite non-trivial solutions."""
    m = len(A)
    n = len(A[0]) if m else 0
    zeros = [Fraction(0) for _ in range(m)]
    res = solve_linear_system(A, zeros)
    if res['unique']:
        res['message'] = 'Sistema homogéneo: solo la solución trivial.'
    else:
        res['message'] = 'Sistema no homogéneo: infinitas soluciones no triviales.'
    return res


def analyze_linear_dependence(vectors: List[List[Fraction]]) -> Dict[str, Any]:
    """
    Given vectors v1..vk in R^n, analyze dependence by solving A c = 0, where A has columns = vectors.
    Returns dict with steps and conclusion.
    """
    if not vectors:
        return {
            'dependent': False,
            'message': 'No se proporcionaron vectores.'
        }
    k = len(vectors)
    n = len(vectors[0])
    # Build A as n x k with columns vectors
    A = [[_to_fraction(0) for _ in range(k)] for _ in range(n)]
    for j, v in enumerate(vectors):
        for i in range(n):
            A[i][j] = _to_fraction(v[i])
    res = solve_homogeneous(A)
    dependent = not res['unique']
    msg = 'Los vectores son linealmente dependientes.' if dependent else 'Los vectores son linealmente independientes.'
    res['dependent'] = dependent
    res['message'] = msg + ' ' + res.get('message', '')
    return res
