# Ecuación matricial AX = B
# A y B son listas de listas (matrices), X es incógnita

def gauss_eliminacion(A, B):
    # A: matriz n x n, B: matriz n x m (o vector n)
    # Devuelve X y pasos
    pasos = []
    n = len(A)
    # Asegurar que B es matriz n x m
    if isinstance(B[0], list):
        m = len(B[0])
        Bmat = B
    else:
        m = 1
        Bmat = [[b] for b in B]
    # Matriz aumentada
    M = [list(A[i]) + list(Bmat[i]) for i in range(n)]
    pasos.append(f"Matriz aumentada inicial: {M}")
    # Eliminación hacia adelante
    for i in range(n):
        # Pivote
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-9:
            pasos.append(f"Sin pivote en columna {i}")
            continue
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]
            pasos.append(f"Intercambio fila {i} con fila {max_row}: {M}")
        piv = M[i][i]
        if abs(piv) > 1e-9:
            M[i] = [x / piv for x in M[i]]
            pasos.append(f"Normalizar fila {i}: {M}")
        for j in range(i+1, n):
            f = M[j][i]
            M[j] = [a - f * b for a, b in zip(M[j], M[i])]
            pasos.append(f"Eliminar fila {j} usando fila {i}: {M}")
    # Sustitución hacia atrás
    X = [[0.0 for _ in range(m)] for _ in range(n)]
    for k in range(m):
        for i in range(n-1, -1, -1):
            if abs(M[i][i]) < 1e-9:
                if abs(M[i][n+k]) > 1e-9:
                    pasos.append("Sistema incompatible")
                    return None, pasos
                else:
                    pasos.append("Infinitas soluciones")
                    return None, pasos
            X[i][k] = M[i][n+k] - sum(M[i][j] * X[j][k] for j in range(i+1, n))
    pasos.append(f"Solución X: {X}")
    return X, pasos
