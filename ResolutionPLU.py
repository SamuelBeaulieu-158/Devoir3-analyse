import numpy as np


def ResolutionPLU(P, L, U, b):

    b = np.array(b, dtype=float)

    n = len(b)

    # Étape 1 : Pb
    Pb = P @ b

    # Étape 2 : substitution avant
    y = np.zeros(n)

    for i in range(n):

        somme = 0

        for j in range(i):
            somme += L[i, j] * y[j]

        y[i] = (Pb[i] - somme) / L[i, i]

    # Étape 3 : substitution arrière
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        somme = 0

        for j in range(i + 1, n):
            somme += U[i, j] * x[j]

        x[i] = (y[i] - somme) / U[i, i]

    return y, x