import numpy as np

def imprimir_tabla(tabla, Z, nombres, bases):
    print("\nTabla:")
    encabezado = nombres + ["RHS"]
    print("Base\t" + "\t".join(encabezado))
    for i, fila in enumerate(tabla):
        base = nombres[bases[i]] if bases[i] < len(nombres) else f"B{bases[i]}"
        print(f"{base}\t" + "\t".join(f"{x:.2f}" for x in fila))
    print("Z\t" + "\t".join(f"{x:.2f}" for x in Z))

def preparar_tabla():
    c = [5, -4, 3]
    A = [
        [2, 1, -1],    
        [1, -3, 2],    
        [1, 1, 1]      
    ]
    b = [10, 5, 15]
    restricciones = ['=', '>=', '<=']

    n_vars = len(c)
    n_rest = len(b)

    nombres = [f'x{i+1}' for i in range(n_vars)]
    tabla = []
    bases = []
    artificiales = []
    col = n_vars

    for i, tipo in enumerate(restricciones):
        fila = A[i].copy()
        extras = [0] * n_rest
        arts = [0] * n_rest

        if tipo == '<=':
            extras[i] = 1
            fila += extras + arts
            nombres.append(f's{i+1}')
            bases.append(col)
            col += 1
        elif tipo == '>=':
            extras[i] = -1
            arts[i] = 1
            fila += extras + arts
            nombres.append(f'e{i+1}')
            nombres.append(f'a{i+1}')
            bases.append(col + 1)
            artificiales.append(col + 1)
            col += 2
        elif tipo == '=':
            arts[i] = 1
            fila += extras + arts
            nombres.append(f'a{i+1}')
            bases.append(col + 1)
            artificiales.append(col + 1)
            col += 1

        fila.append(b[i])
        tabla.append(fila)

    tabla = np.array(tabla, dtype=float)
    return c, tabla, bases, artificiales, nombres

def construir_Z(tabla, bases, artificiales):
    Z = np.zeros(tabla.shape[1])
    for i, b in enumerate(bases):
        if b in artificiales:
            Z -= tabla[i]
    return Z

def pivotear(tabla, Z, fila, col, bases):
    tabla[fila] /= tabla[fila, col]
    for i in range(len(tabla)):
        if i != fila:
            tabla[i] -= tabla[i, col] * tabla[fila]
    Z -= Z[col] * tabla[fila]
    bases[fila] = col

def simplex(tabla, Z, bases, nombres, fase=1):
    while True:
        if all(Z[:-1] >= -1e-8):
            break
        col = np.argmin(Z[:-1])
        if all(tabla[:, col] <= 0):
            raise ValueError("Problema no acotado.")
        ratios = [fila[-1] / fila[col] if fila[col] > 0 else np.inf for fila in tabla]
        fila = np.argmin(ratios)
        pivotear(tabla, Z, fila, col, bases)
    return tabla, Z, bases

def eliminar_artificiales(tabla, Z, nombres, artificiales):
    tabla = np.delete(tabla, artificiales, axis=1)
    Z = np.delete(Z, artificiales)
    for i in sorted(artificiales, reverse=True):
        nombres.pop(i)
    return tabla, Z, nombres

def simplex_dos_fases():
    c, tabla, bases, artificiales, nombres = preparar_tabla()

    # Fase I
    Z = construir_Z(tabla, bases, artificiales)
    tabla, Z, bases = simplex(tabla, Z, bases, nombres, fase=1)

    if abs(Z[-1]) > 1e-5:
        raise ValueError("Problema infactible.")

    # Eliminar artificiales
    tabla, Z, nombres = eliminar_artificiales(tabla, Z, nombres, artificiales)

    # Fase II
    c_ext = [-ci for ci in c] + [0]*(len(tabla[0])-len(c)-1)
    Z = np.zeros(len(c_ext)+1)
    Z[:-1] = c_ext
    for i, b in enumerate(bases):
        Z -= Z[b] * tabla[i]

    tabla, Z, bases = simplex(tabla, Z, bases, nombres, fase=2)

    solucion = np.zeros(len(c))
    for i, b in enumerate(bases):
        if b < len(c):
            solucion[b] = tabla[i, -1]

    valor_optimo = sum(c[i] * solucion[i] for i in range(len(c)))
    return solucion, valor_optimo

# Ejecutar
solucion, valor = simplex_dos_fases()
print("\nSolución óptima:", solucion)
print("Valor mínimo de Z:", valor)