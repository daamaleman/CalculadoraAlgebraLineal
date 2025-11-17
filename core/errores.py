"""Módulo `core.errores` — Herramientas didácticas sobre representación posicional,
tipos de error numérico y ejemplos con punto flotante.

Contiene:
- descomponer números en base 10 y base 2 mostrando pasos.
- explicaciones y pequeñas demostraciones de errores (inherente, redondeo,
  truncamiento, overflow/underflow, error del modelo).
- ejemplos de punto flotante (incluye la prueba 0.1 + 0.2 == 0.3).
- demo opcional con NumPy si está disponible.
- funciones para calcular error absoluto, relativo y propagación de error para
  f(x) = sin(x) + x^2.

Este módulo se puede ejecutar como script para una experiencia guiada por
menú en consola.
"""

from math import sin, cos
from typing import Tuple

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def decompose_base10(value) -> str:
    """Descompone un número (entero o con parte fraccionaria) en base 10 mostrando
    cada cifra multiplicada por su potencia correspondiente (incluye potencias negativas
    para la parte decimal).

    value puede ser int, float o str. Retorna una cadena con los pasos.
    """
    s = str(value).strip()
    if not s:
        return "Entrada vacía"
    # Manejar signo
    sign = ''
    if s[0] == '-':
        sign = '-'
        s = s[1:]

    # Separar parte entera y fraccionaria
    if '.' in s:
        int_part, frac_part = s.split('.', 1)
    else:
        int_part, frac_part = s, ''

    # Validación básica: todas las partes deben ser dígitos
    if (not int_part.isdigit()) or (frac_part and not frac_part.isdigit()):
        return "Entrada inválida: asegúrese de ingresar un número decimal válido (ej. 84506 o 12.345)."

    lines = []
    total = 0.0

    # Parte entera
    if int_part:
        m = len(int_part)
        parts = []
        for i, ch in enumerate(int_part):
            digit = int(ch)
            power = m - i - 1
            term = digit * (10 ** power)
            lines.append(f"{digit} × 10^{power} = {digit} × {10**power} = {term}")
            total += term
            parts.append(f"{digit}×10^{power}")
        lines.append(f"Suma parte entera = {' + '.join(parts)}")

    # Parte fraccionaria (potencias negativas)
    if frac_part:
        parts = []
        for j, ch in enumerate(frac_part, start=1):
            digit = int(ch)
            power = -j
            term = digit * (10 ** power)
            # Mostrar 10**power como 10^-j y el valor decimal con repr
            lines.append(f"{digit} × 10^{power} = {digit} × 10^{power} = {term}")
            total += term
            parts.append(f"{digit}×10^{power}")
        lines.append(f"Suma parte fraccionaria = {' + '.join(parts)}")

    lines.append(f"Suma final = {sign}{format(total, '.16g')}")
    return "\n".join(lines)


def decompose_base2(bstr: str) -> str:
    """Descompone una cadena de bits (ej. '1111001') en potencias de 2 y suma final.

    Retorna una cadena con los pasos. Ignora espacios.
    """
    s = bstr.strip().replace(' ', '')
    if any(c not in '01' for c in s):
        return "Entrada inválida: la cadena debe contener solo 0 o 1."
    m = len(s)
    lines = []
    total = 0
    for i, ch in enumerate(s):
        bit = int(ch)
        power = m - i - 1
        term = bit * (2 ** power)
        lines.append(f"{bit} × 2^{power} = {term}")
        total += term
    lines.append(f"Suma final = {' + '.join([f'{int(ch)}×2^{len(s)-i-1}' for i,ch in enumerate(s)])}")
    lines.append(f"Resultado decimal = {total}")
    return "\n".join(lines)


def explain_errors() -> str:
    """Devuelve una explicación concisa de distintos tipos de error con pequeños ejemplos."""
    lines = []
    lines.append("Error inherente: es el error en los datos/mediciones. Ej.: medir longitud con regla de 1 mm -> incertidumbre ±0.5 mm")
    lines.append("")
    lines.append("Error de redondeo: ocurre al representar un número con una cantidad finita de dígitos. Ej.: round(2.675, 2) puede dar 2.67 en IEEE-754 en Python debido a la representación interna.")
    lines.append("")
    lines.append("Error de truncamiento: se produce al aproximar un proceso infinito por uno finito. Ej.: aproximar sin(x) ≈ x - x^3/6 (Taylor de orden 3) genera truncamiento.")
    lines.append("")
    lines.append("Overflow / Underflow: overflow -> valor demasiado grande -> inf; underflow -> valor muy pequeño representado como 0. Ejemplo con NumPy si disponible.")
    lines.append("")
    lines.append("Error del modelo: la discrepancia por cómo modelamos un problema (p. ej. usar una ecuación lineal para un fenómeno no lineal).")
    return "\n".join(lines)


def float_examples() -> str:
    """Muestra ejemplos de comportamiento de punto flotante y explica por qué 0.1+0.2 != 0.3 exactamente."""
    out = []
    out.append("Ejemplo clásico en Python:")
    eq = (0.1 + 0.2 == 0.3)
    out.append(f"0.1 + 0.2 == 0.3 -> {eq}")
    out.append("")
    out.append("Explicación: 0.1 y 0.2 no tienen representación exacta en binario finito (base 2), por lo que las sumas producen pequeñas fracciones residuales. La comparación de igualdad falla porque las representaciones aproximadas no suman exactamente 0.3.")
    out.append("")
    out.append(f"Representaciones internas aproximadas: 0.1 -> {format(0.1,'.17g')}, 0.2 -> {format(0.2,'.17g')}, 0.1+0.2 -> {format(0.1+0.2, '.17g')}")
    return "\n".join(out)


def numpy_demo() -> str:
    """Demuestra algunas diferencias con NumPy si está disponible."""
    if not _HAS_NUMPY:
        return "NumPy no está disponible en este entorno. Instale numpy para ver la demo."
    out = []
    out.append("NumPy demo: tipos float64 vs float32")
    a64 = np.float64(0.1)
    b64 = np.float64(0.2)
    out.append(f"float64: 0.1 + 0.2 == {a64 + b64}")
    a32 = np.float32(0.1)
    b32 = np.float32(0.2)
    out.append(f"float32: 0.1 + 0.2 == {a32 + b32}")
    out.append("")
    out.append("Ejemplo overflow/underflow con NumPy:")
    big = np.float64(1e308)
    try:
        of = big * big
        out.append(f"1e308 * 1e308 -> {of}")
    except Exception as e:
        out.append(f"Overflow demo error: {e}")
    tiny = np.float64(1e-324)
    out.append(f"1e-324 representado en float64 -> {tiny}")
    return "\n".join(out)


def absolute_error(true: float, approx: float) -> float:
    """Error absoluto E_a = |x_v - x_a|"""
    return abs(true - approx)


def relative_error(true: float, approx: float) -> float:
    """Error relativo E_r = |x_v - x_a| / |x_v|, devuelve None si true == 0 (no definido)."""
    if true == 0:
        return None
    return abs(true - approx) / abs(true)


def error_propagation_fx(true_x: float, approx_x: float) -> Tuple[float, float, float, float]:
    """Para f(x) = sin(x) + x^2 calcula:
    - f(true_x)
    - f(approx_x)
    - estimación de propagación del error: |f'(x_true)| * E_a(x)
    Retorna (f_true, f_approx, E_prop, f_diff)

    Nota: usamos la derivada f'(x) = cos(x) + 2x. La propagación lineal es una cota lineal de la variación.
    """
    f_true = sin(true_x) + true_x ** 2
    f_approx = sin(approx_x) + approx_x ** 2
    Ea_x = absolute_error(true_x, approx_x)
    deriv = abs(cos(true_x) + 2 * true_x)
    E_prop = deriv * Ea_x
    f_diff = abs(f_true - f_approx)
    return f_true, f_approx, E_prop, f_diff


def format_table_error(true_x: float, approx_x: float) -> str:
    """Construye una tabla legible con E_a, E_r y propagación para f(x)."""
    Ea_x = absolute_error(true_x, approx_x)
    Er_x = relative_error(true_x, approx_x)
    f_true, f_approx, E_prop, f_diff = error_propagation_fx(true_x, approx_x)
    lines = []
    lines.append("+----------------------+---------------------------+")
    lines.append("| Cantidad             | Valor                     |")
    lines.append("+----------------------+---------------------------+")
    lines.append(f"| x verdadero (x_v)    | {true_x:>25.16g} |")
    lines.append(f"| x aproximado (x_a)   | {approx_x:>25.16g} |")
    lines.append(f"| Error absoluto E_a   | {Ea_x:>25.16g} |")
    lines.append(f"| Error relativo E_r   | {('N/A' if Er_x is None else format(Er_x, '.8g')):>25} |")
    lines.append(f"| f(x_v)               | {f_true:>25.16g} |")
    lines.append(f"| f(x_a)               | {f_approx:>25.16g} |")
    lines.append(f"| Propagación ≈ |f'(x_v)|·E_a | {E_prop:>25.16g} |")
    lines.append(f"| Diferencia exacta | |f(x_v)-f(x_a)| | {f_diff:>25.16g} |")
    lines.append("+----------------------+---------------------------+")
    lines.append("")
    lines.append("Interpretación breve:")
    lines.append("- E_a mide la desviación absoluta en la entrada x.")
    lines.append("- E_r es útil para conocer la magnitud relativa (si x_v ≠ 0).")
    lines.append("- La propagación lineal usa la derivada para estimar cuánto cambia f si x cambia en E_a. Si la diferencia exacta es mayor que la propagación, existen efectos no lineales o errores numéricos adicionales.")
    return "\n".join(lines)


def _input_float(prompt: str) -> float:
    while True:
        try:
            s = input(prompt)
            return float(s)
        except Exception:
            print("Entrada inválida. Intente de nuevo (ejemplo: 1.23, 0.5, -2).")


def main_menu() -> None:
    print("Módulo de errores numéricos — menú interactivo")
    while True:
        print("\nOpciones:")
        print("1) Descomponer en base 10 (enteros)")
        print("2) Descomponer en base 2 (bits)")
        print("3) Explicación de tipos de error")
        print("4) Ejemplos de punto flotante (0.1+0.2)")
        print("5) Demo NumPy (si está instalado)")
        print("6) Ejercicio principal: calcular errores y propagación para f(x)=sin(x)+x^2")
        print("0) Salir")
        choice = input("Elija una opción: ")
        if choice == '0':
            print("Saliendo...")
            break
        elif choice == '1':
            s = input("Ingrese un entero no negativo (ej. 84506): ")
            try:
                n = int(s)
                print(decompose_base10(n))
            except Exception:
                print("Entrada inválida: ingrese un entero.")
        elif choice == '2':
            s = input("Ingrese cadena de bits (ej. 1111001): ")
            print(decompose_base2(s))
        elif choice == '3':
            print(explain_errors())
        elif choice == '4':
            print(float_examples())
        elif choice == '5':
            print(numpy_demo())
        elif choice == '6':
            print("Ingrese el valor verdadero x_v:")
            xv = _input_float("x_v = ")
            print("Ingrese el valor aproximado x_a:")
            xa = _input_float("x_a = ")
            print(format_table_error(xv, xa))
        else:
            print("Opción no reconocida. Intente de nuevo.")


if __name__ == '__main__':
    main_menu()
