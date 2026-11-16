import numpy as np

from DecompositionPLU import DecompositionPLU
from ResolutionPLU import ResolutionPLU

# Constantes de l'énoncé
a = 0.866025404
b = 0.707106781

A = np.array([

    [-a, 0, b, 1, 0, 0, 0, 0, 0, 0, 0],
    [-0.5, 0, -b, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, -b, 0, b, 1, 0, 0, 0, 0, 0],
    [0, 0, b, 0, b, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, -b, 0, b, 1, 0, 0, 0],
    [0, 0, 0, 0, -b, 0, -b, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, -b, 0, b, 1, 0],
    [0, 0, 0, 0, 0, 0, b, 0, b, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1, -b, 0, a],
    [0, 0, 0, 0, 0, 0, 0, 0, -b, 0, -0.5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -a]

], dtype=float)

# VECTEUR b

vecteur_b = np.array([
    0,
    0,
    0,
    10000,
    0,
    0,
    0,
    20000,
    0,
    0,
    0
], dtype=float)

# FACTORISATION PLU (DOOLITTLE)

P, L, U = DecompositionPLU(A, "Doolittle")

# RÉSOLUTION

y, T = ResolutionPLU(P, L, U, vecteur_b)

# AFFICHAGE DES TENSIONS

print("\nTensions obtenues :\n")

for i in range(len(T)):
    print(f"T{i+1} = {T[i]:.5f} N")


n = A.shape[0]
A_inverse = np.zeros((n, n)) 
for i in range(n):
    e = np.zeros(n) 
    e[i] = 1.0
    y, x = ResolutionPLU(P, L, U, e) 
    A_inverse[:, i] = x 
print("\nInverse de A :\n") 
print(A_inverse)

b_etoile = vecteur_b.copy()

b_etoile[3] += 7e-6
b_etoile[7] += 2e-5

delta_b = b_etoile - vecteur_b

# Normes de b*
norme_betoile_1 = np.linalg.norm(b_etoile, 1)
norme_betoile_2 = np.linalg.norm(b_etoile, 2)
norme_betoile_inf = np.linalg.norm(b_etoile, np.inf)

# Normes de (b - b*)
norme_delta_b_1 = np.linalg.norm(delta_b, 1)
norme_delta_b_2 = np.linalg.norm(delta_b, 2)
norme_delta_b_inf = np.linalg.norm(delta_b, np.inf)


print("\nNormes de b*")
print("||b*||1 =", norme_betoile_1)
print("||b*||2 =", norme_betoile_2)
print("||b*||∞ =", norme_betoile_inf)

print("\nNormes de (b - b*)")
print("||b - b*||1 =", norme_delta_b_1)
print("||b - b*||2 =", norme_delta_b_2)
print("||b - b*||∞ =", norme_delta_b_inf)

# Normes de A
normeA1 = np.linalg.norm(A, 1)
normeA2 = np.linalg.norm(A, 2)
normeAinf = np.linalg.norm(A, np.inf)

# Normes de A^-1
normeInv1 = np.linalg.norm(A_inverse, 1)
normeInv2 = np.linalg.norm(A_inverse, 2)
normeInvInf = np.linalg.norm(A_inverse, np.inf)

# Nombres de conditionnement
cond1 = normeA1 * normeInv1
cond2 = normeA2 * normeInv2
condInf = normeAinf * normeInvInf

# Rapport ||Δb|| / ||b*||
rapport1 = np.linalg.norm(delta_b, 1) / np.linalg.norm(b_etoile, 1)
rapport2 = np.linalg.norm(delta_b, 2) / np.linalg.norm(b_etoile, 2)
rapportInf = np.linalg.norm(delta_b, np.inf) / np.linalg.norm(b_etoile, np.inf)

# Intervalles d'erreurs relatives

borne_inf_1 = rapport1 / cond1
borne_sup_1 = rapport1 * cond1

borne_inf_2 = rapport2 / cond2
borne_sup_2 = rapport2 * cond2

borne_inf_inf = rapportInf / condInf
borne_sup_inf = rapportInf * condInf


print("\nNormes de A")
print("||A||1 =", normeA1)
print("||A||2 =", normeA2)
print("||A||∞ =", normeAinf)

print("\nNormes de A^-1")
print("||A^-1||1 =", normeInv1)
print("||A^-1||2 =", normeInv2)
print("||A^-1||∞ =", normeInvInf)


print("\nErreurs relatifs")
print("||b-b*||1 / ||b*||1 =", rapport1)
print("||b-b*||2 / ||b*||2 =", rapport2)
print("||b-b*||∞ / ||b*||∞ =", rapportInf)

print("\ncond1 =", cond1)
print("cond2 =", cond2)
print("condInf =", condInf)

print("\nNorme 1 :")
print("[", borne_inf_1, ",", borne_sup_1, "]")

print("\nNorme 2 :")
print("[", borne_inf_2, ",", borne_sup_2, "]")

print("\nNorme inf :")
print("[", borne_inf_inf, ",", borne_sup_inf, "]")