from core.vector import Vector

# --- Combinación lineal ---
def es_combinacion_lineal(vectores, objetivo):
    # vectores: lista de Vector
    # objetivo: Vector
    n = len(vectores)
    m = len(objetivo.values)
    # Construir sistema de ecuaciones Ax = b
    A = [v.values[:] for v in vectores]
    A = list(map(list, zip(*A)))  # columnas = vectores
    b = objetivo.values[:]
    # Resolver sistema
    solucion, pasos = resolver_sistema(A, b)
    es_comb = solucion is not None
    return es_comb, pasos

# --- Resolución de sistemas por Gauss ---
def resolver_sistema(A, b):
    # A: matriz (lista de listas), b: vector (lista)
    pasos = []
    n = len(A)
    m = len(A[0])
    # Construir matriz aumentada
    M = [A[i][:] + [b[i]] for i in range(n)]
    pasos.append(f"Matriz aumentada inicial: {M}")
    # Eliminación hacia adelante
    for i in range(min(n, m)):
        # Buscar pivote
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-9:
            continue
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            pasos.append(f"Intercambio fila {i} con fila {max_row}: {M}")
        # Hacer pivote 1
        piv = M[i][i]
        if abs(piv) > 1e-9:
            M[i] = [x / piv for x in M[i]]
            pasos.append(f"Normalizar fila {i}: {M}")
        # Eliminar debajo
        for j in range(i+1, n):
            f = M[j][i]
            M[j] = [a - f * b for a, b in zip(M[j], M[i])]
            pasos.append(f"Eliminar fila {j} usando fila {i}: {M}")
    # Sustitución hacia atrás
    x = [0] * m
    for i in range(m-1, -1, -1):
        if i >= n or abs(M[i][i]) < 1e-9:
            if abs(M[i][-1]) > 1e-9:
                pasos.append("Sistema incompatible")
                return None, pasos
            else:
                pasos.append("Infinitas soluciones")
                return None, pasos
        x[i] = M[i][-1] - sum(M[i][j] * x[j] for j in range(i+1, m))
    pasos.append(f"Solución: {x}")
    return x, pasos

# --- Ecuación vectorial ---
def resolver_ecuacion_vectorial(vectores, objetivo):
    # Plantea y resuelve c1*v1 + ... + cn*vn = objetivo
    return es_combinacion_lineal(vectores, objetivo)
