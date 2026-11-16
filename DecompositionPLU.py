import numpy as np

def DecompositionPLU(A, choix_decomposition="Doolittle"):

    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("La matrice doit être carrée.")

    choix = choix_decomposition.lower()

    # CHOLESKI
    if choix == "choleski":

        # Vérification de symétrie
        if not np.allclose(A, A.T):
            return "La matrice ne possède pas de factorisation de Choleski"

        try:
            L = np.linalg.cholesky(A)
            U = L.T
            P = np.eye(n)

            return P, L, U

        except np.linalg.LinAlgError:
            return "La matrice ne possède pas de factorisation de Choleski"

    # DOOLITTLE ET CROUT
    P = np.eye(n)
    L = np.zeros((n, n))
    U = A.copy()

    for k in range(n):

        pivot = np.argmax(np.abs(U[k:n, k])) + k

        if U[pivot, k] == 0:
            raise ValueError("La matrice est singulière.")

        if pivot != k:

            U[[k, pivot], :] = U[[pivot, k], :]
            P[[k, pivot], :] = P[[pivot, k], :]

            if k > 0:
                L[[k, pivot], :k] = L[[pivot, k], :k]

        for i in range(k + 1, n):

            facteur = U[i, k] / U[k, k]

            L[i, k] = facteur

            U[i, :] = U[i, :] - facteur * U[k, :]

    np.fill_diagonal(L, 1)

    # DOOLITTLE
    if choix == "doolittle":
        return P, L, U

    # CROUT
    elif choix == "crout":

        D = np.diag(np.diag(U))

        U_crout = np.linalg.inv(D) @ U
        L_crout = L @ D

        return P, L_crout, U_crout

    else:
        raise ValueError(
            "choix_decomposition doit être "
            "'Doolittle', 'Crout' ou 'Choleski'"
        )