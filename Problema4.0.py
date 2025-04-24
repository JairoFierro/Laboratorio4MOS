# Jairo Fierro y Juan Felipe Puig

import numpy as np

def build_tableau(c1=4, c2=3, b1=8, b2=12):

    Z = [-c1, -c2, 0, 0, 0]

    R1 = [1, 2, 1, 0, b1]
    R2 = [3, 2, 0, 1, b2]
    return np.array([Z, R1, R2], dtype=float)

def pivot(tableau, row, col):
    pivot_element = tableau[row, col]
    tableau[row] /= pivot_element
    for r in range(len(tableau)):
        if r != row:
            tableau[r] -= tableau[r, col] * tableau[row]

def simplex(tableau, var_names=["x1", "x2", "s1", "s2"]):
    m, n = tableau.shape
    basis = ["s1", "s2"]
    while any(tableau[0, :-1] < 0):
        pivot_col = np.argmin(tableau[0, :-1])
        if all(tableau[1:, pivot_col] <= 0):
            raise Exception("Problema no acotado.")
        ratios = [
            tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 0 else np.inf
            for i in range(1, m)
        ]
        pivot_row = 1 + np.argmin(ratios)
        pivot(tableau, pivot_row, pivot_col)
        basis[pivot_row - 1] = var_names[pivot_col]
    Z = tableau[0, -1]
    solution = dict.fromkeys(var_names, 0)
    for i, var in enumerate(basis):
        col = var_names.index(var)
        solution[var] = tableau[i+1, -1]
    return solution, Z, basis, tableau

# Solución del problema original
tableau = build_tableau()
solution, Z, basis, final_tableau = simplex(tableau)

print("Solución óptima:")
print("Z =", Z)
for var in ["x1", "x2"]:
    print(f"{var} =", solution[var])
print("Base óptima:", basis)

# Precios sombra (multiplicadores duales)
# Son los coeficientes de las variables de holgura en la fila Z
print("Precios sombra:")
print("y1 =", final_tableau[0, 2])  # s1
print("y2 =", final_tableau[0, 3])  # s2
