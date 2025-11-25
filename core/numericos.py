"""Métodos numéricos: Bisección y Regla Falsa.

Incluye:
- `parse_function(expr: str) -> callable` para evaluar f(x) de forma segura usando `math`.
- `biseccion(f, a, b, tol, max_iter=100)` ejecuta el método de bisección.
- `regla_falsa(f, a, b, tol, max_iter=100)` ejecuta el método de falsa posición.

Cada método devuelve un diccionario con:
- root: raíz aproximada
- iterations: lista de iteraciones con columnas (iter, a, b, x, f(x), error_abs, error_rel_percent, intervalo)
- final_error_abs, final_error_percent
- interval: intervalo final que encierra la raíz
- converged: bool
- message: texto de estado

Notas:
- La condición inicial f(a)*f(b) < 0 se verifica; si no se cumple, se lanza ValueError.
- La tolerancia `tol` se interpreta como criterio de parada sobre el error absoluto |x_k - x_{k-1}|.
- Se muestra también el error relativo porcentual para cada paso cuando está definido.
"""

from math import *
from typing import Callable, Dict, Any


def parse_function(expr: str) -> Callable[[float], float]:
    """Construye una función f(x) a partir de una cadena `expr`.

    - Soporta funciones de `math` (sin, cos, exp, log, sqrt, etc.).
    - Reemplaza '^' por '**' para exponentes.
    - Expone constantes `pi` y `e`.
    - Evalúa con entorno seguro (sin builtins).
    """
    expr = expr.strip()
    expr = expr.replace('^', '**')
    # Entorno seguro solo con símbolos de math y constantes útiles
    safe_env: Dict[str, Any] = {name: globals()[name] for name in (
        'sin','cos','tan','asin','acos','atan','exp','log','log10','sqrt','fabs','floor','ceil','pow','pi','e'
    )}
    # También permitir abs y pow del entorno
    safe_env['abs'] = abs
    safe_env['pow'] = pow

    def f(x: float) -> float:
        return eval(expr, {"__builtins__": {}}, {**safe_env, 'x': x})

    return f


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def biseccion(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int = 100) -> Dict[str, Any]:
    fa = f(a); fb = f(b)
    if fa * fb >= 0:
        raise ValueError('El intervalo no es válido: f(a)·f(b) ≥ 0')
    iterations = []
    prev_x = None
    converged = False
    message = 'OK'
    for k in range(1, max_iter + 1):
        x = (a + b) / 2.0
        fx = f(x)

        # Error absoluto y relativo porcentual
        if prev_x is None:
            err_abs = None
            err_rel_pct = None
        else:
            err_abs = abs(x - prev_x)
            err_rel_pct = None if x == 0 else (err_abs / abs(x)) * 100.0

        # Registrar fila
        iterations.append({
            'iter': k,
            'a': a,
            'b': b,
            'x': x,
            'fx': fx,
            'error_abs': err_abs,
            'error_rel_percent': err_rel_pct,
            'interval': (a, b),
        })

        # Criterio de parada por tolerancia en el error absoluto
        if err_abs is not None and err_abs < tol:
            converged = True
            message = 'Convergió por tolerancia'
            prev_x = x
            break

        # Actualizar intervalo por cambio de signo
        if fa * fx < 0:
            b = x; fb = fx
        else:
            a = x; fa = fx

        prev_x = x

    # Resultado final
    root = prev_x if prev_x is not None else (a + b) / 2.0
    final_err_abs = None
    final_err_pct = None
    if len(iterations) >= 2:
        x_now = iterations[-1]['x']
        x_prev = iterations[-2]['x']
        final_err_abs = abs(x_now - x_prev)
        final_err_pct = None if x_now == 0 else (final_err_abs / abs(x_now)) * 100.0
    return {
        'root': root,
        'iterations': iterations,
        'final_error_abs': final_err_abs,
        'final_error_percent': final_err_pct,
        'interval': (a, b),
        'converged': converged,
        'message': message,
    }


def regla_falsa(f: Callable[[float], float], a: float, b: float, tol: float, max_iter: int = 100) -> Dict[str, Any]:
    fa = f(a); fb = f(b)
    if fa * fb >= 0:
        raise ValueError('El intervalo no es válido: f(a)·f(b) ≥ 0')
    iterations = []
    prev_x = None
    converged = False
    message = 'OK'
    for k in range(1, max_iter + 1):
        # Punto de intersección (falsa posición)
        x = b - fb * (b - a) / (fb - fa)
        fx = f(x)

        if prev_x is None:
            err_abs = None
            err_rel_pct = None
        else:
            err_abs = abs(x - prev_x)
            err_rel_pct = None if x == 0 else (err_abs / abs(x)) * 100.0

        iterations.append({
            'iter': k,
            'a': a,
            'b': b,
            'x': x,
            'fx': fx,
            'error_abs': err_abs,
            'error_rel_percent': err_rel_pct,
            'interval': (a, b),
        })

        if err_abs is not None and err_abs < tol:
            converged = True
            message = 'Convergió por tolerancia'
            prev_x = x
            break

        # Actualización del intervalo según cambio de signo
        if fa * fx < 0:
            b = x; fb = fx
        else:
            a = x; fa = fx

        prev_x = x

    root = prev_x if prev_x is not None else b - fb * (b - a) / (fb - fa)
    final_err_abs = None
    final_err_pct = None
    if len(iterations) >= 2:
        x_now = iterations[-1]['x']
        x_prev = iterations[-2]['x']
        final_err_abs = abs(x_now - x_prev)
        final_err_pct = None if x_now == 0 else (final_err_abs / abs(x_now)) * 100.0
    return {
        'root': root,
        'iterations': iterations,
        'final_error_abs': final_err_abs,
        'final_error_percent': final_err_pct,
        'interval': (a, b),
        'converged': converged,
        'message': message,
    }


def format_iterations_table(result: Dict[str, Any], method_name: str) -> str:
    """Devuelve una tabla de texto monoespaciado con las iteraciones y un resumen final."""
    lines = []
    lines.append(f"Método: {method_name}")
    lines.append("+-----+-----------+-----------+-----------+-----------+-----------+-----------+----------------+")
    lines.append("| it  | a         | b         | x         | f(x)      | err_abs   | err_rel%  | intervalo      |")
    lines.append("+-----+-----------+-----------+-----------+-----------+-----------+-----------+----------------+")
    for row in result['iterations']:
        it = row['iter']
        a = row['a']; b = row['b']; x = row['x']; fx = row['fx']
        ea = row['error_abs']; er = row['error_rel_percent']
        interval = row['interval']
        fmt = lambda v: ("" if v is None else f"{v:.6g}")
        lines.append(f"| {it:<3d} | {a:>9.6g} | {b:>9.6g} | {x:>9.6g} | {fx:>9.6g} | {fmt(ea):>9} | {fmt(er):>9} | ({interval[0]:.6g},{interval[1]:.6g}) |")
    lines.append("+-----+-----------+-----------+-----------+-----------+-----------+-----------+----------------+")
    root = result['root']
    ferr_abs = result['final_error_abs']
    ferr_pct = result['final_error_percent']
    interval = result['interval']
    lines.append(f"Raíz aproximada: {root:.10g}")
    lines.append(f"Iteraciones totales: {len(result['iterations'])}")
    lines.append(f"Error final abs: {'' if ferr_abs is None else f'{ferr_abs:.6g}'}")
    lines.append(f"Error final %: {'' if ferr_pct is None else f'{ferr_pct:.6g}'}")
    lines.append(f"Intervalo final: ({interval[0]:.6g}, {interval[1]:.6g})")
    lines.append(f"Estado: {'Convergió' if result['converged'] else 'Sin converger'} — {result['message']}")
    return "\n".join(lines)
