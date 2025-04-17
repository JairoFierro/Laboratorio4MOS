import numpy as np
import sympy as sp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyomo.environ import *

import time


class ProblemaTres:
    def __init__(self, n=10,m=8,
                 solver_name='glpk'):

        # Parametros para el modelo
        self.solver_name = solver_name
        self.n = n
        self.m = m

        # Placeholders for Pyomo objects
        self.model = None
        self.results = None
        self.c = [5, 8, 3, 7, 6, 9, 4, 10, 2, 11]
        self.b = [50, 60, 55, 40, 45, 70, 65, 50]
        self.A = [
            [1, 2, 1, 1, 0, 0, 3, 1, 2, 1],
            [2, 1, 0, 2, 1, 1, 0, 3, 1, 2],  
            [1, 1, 2, 0, 2, 1, 1, 0, 3, 1],  
            [0, 2, 1, 1, 1, 0, 2, 1, 1, 1],  
            [2, 0, 1, 1, 1, 2, 1, 1, 0, 2],  
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
            [0, 1, 2, 1, 0, 1, 2, 1, 1, 0],  
            [1, 0, 1, 2, 1, 0, 1, 2, 1, 1]   
        ]
        # Variables para el método simplex

    

    def setup_model(self):


        self.model = ConcreteModel()

        self.model.I = RangeSet(1, self.n)
        self.model.J = RangeSet(1, self.m)

        # Variable de decisión
        self.model.x = Var(self.model.I, domain=NonNegativeReals)

        #Definir función objetivo 
        self.model.obj = Objective(expr=sum(self.c[i-1] * self.model.x[i] for i in self.model.I), sense=maximize)

        # Restricciones
        def restricciones(model, j):
            return sum(self.A[j-1][i-1] * model.x[i] for i in model.I) <= self.b[j-1]

        self.model.restricciones = Constraint(self.model.J, rule=restricciones)

    def verificar_optimalidad(self,tabla_simplex):

        # Seleccionamos la fila que es la funcion objetivo
        fila_z = tabla_simplex[-1, :-1]  # todos menos RHS

        print("\nCostos reducidos :")
        print(fila_z)

        # Verificar si todos los c̄j >= 0
        if np.all(fila_z >= 0):
            return True
        else:
            return False




    # El tercer paso es seleccionar variable de entrada

    def seleccionar_variable_entrada(self,tabla_simplex):
        fila_z = tabla_simplex[-1, :-1] 

        # Verificar si hay alguna variable negativa
        if np.all(fila_z >= 0):
            return None

        # Regla de Dantzig para devolver el más negativo 
        # argmin devuelve el índice mínimo dentro de un arreglo NumPy
        col_entrada = np.argmin(fila_z)
        valor = fila_z[col_entrada]

        return col_entrada


    # El cuarto paso es calcular las razones

    def calcular_variable_salida(self,tabla_simplex, col_entrada):
        # No usar la fila Z
        num_filas = tabla_simplex.shape[0] - 1   
        rhs = tabla_simplex[:num_filas, -1]   

        # Columna de la variable que entra   
        columna = tabla_simplex[:num_filas, col_entrada]  

        razones = []

        for i in range(num_filas):
            a = columna[i]
            b = rhs[i]
            if a > 0:
                theta = b / a
                razones.append((i, theta))

        # Si todos los coeficientes aij <= 0, el problema es ilimitado. Terminar.
        if not razones:
            return None  


        # El paso 5 es seleccionar la variable de salida
        fila_salida = min(razones, key=lambda x: x[1])[0]
        return fila_salida


    # El sexto paso es actualizar la tabla (operación de pivoteo)

    def operacion_pivoteo(self,tabla_simplex, fila_salida, col_entrada, variables_basicas):
        tabla = tabla_simplex.copy().astype(float)

        pivote = tabla[fila_salida, col_entrada]

        # Dividir la fila pivote por el valor del pivote
        tabla[fila_salida, :] = tabla[fila_salida, :] / pivote

        # Para cada fila distinta de la fila pivote, eliminar el valor en la columna pivote
        num_filas = tabla.shape[0]
        for i in range(num_filas):
            if i != fila_salida:
                factor = tabla[i, col_entrada]
                tabla[i, :] -= factor * tabla[fila_salida, :]

        # Actualizar las variables básicas
        nueva_variable = f"x{col_entrada + 1}" if col_entrada < 3 else f"s{col_entrada - 2}"

        variables_basicas[fila_salida] = nueva_variable

        return tabla, variables_basicas


    # Esta función ejecuta el metodo simplex y muestra cada iteración del algoritmo
    def simplex(self,tabla_simplex, variables_basicas):
        iteracion = 1
        while True:
            print("\n Iteración " + str(iteracion))
            if self.verificar_optimalidad(tabla_simplex):
                print("\nSolución óptima.")
                break

            col_entrada = self.seleccionar_variable_entrada(tabla_simplex)
            if col_entrada is None:
                print("No hay variable válida de entrada.")
                break

            fila_salida = self.calcular_variable_salida(tabla_simplex, col_entrada)
            if fila_salida is None:
                print("El problema es ilimitado.")
                break

            tabla_simplex, variables_basicas = self.operacion_pivoteo(
                tabla_simplex, fila_salida, col_entrada, variables_basicas
            )
            iteracion += 1

        return tabla_simplex, variables_basicas
    
    def solve_with_simplex(self):
        A = np.array(self.A)
        b = np.array(self.b).reshape(-1, 1)
        c = np.array(self.c)

        n_vars = len(c)
        n_restr = len(b)

        # Crear matriz aumentada A|I para variables de holgura
        identidad = np.identity(n_restr)
        A_aug = np.hstack((A, identidad))

        # Coeficientes extendidos de la función objetivo
        c_aug = np.append(-c, [0]*n_restr)

        # Construir tabla inicial
        tabla = np.hstack((A_aug, b))
        fila_z = np.append(c_aug, [0])
        tabla = np.vstack((tabla, fila_z))

        # Variables básicas iniciales
        variables_basicas = [f"s{i+1}" for i in range(n_restr)]

        # Ejecutar Simplex
        tabla_final, variables_finales = self.simplex(tabla, variables_basicas)

        # Mostrar solución
        print("\nResultado con Simplex Estándar:")
        for i, var in enumerate(variables_finales):
            print(f"{var} = {tabla_final[i, -1]:.2f}")
        print(f"Z = {tabla_final[-1, -1]:.2f}")

    def solve_model(self):

        if self.model is None:
            raise ValueError("Model is not set up. Please call setup_model() first")
        
        solver = SolverFactory(self.solver_name)
        logfile_path = "glpk_output.log"

        # Ejecutar el solver y guardar la salida
        self.results = solver.solve(self.model, tee=True, logfile=logfile_path)

        # Leer el archivo de log y contar iteraciones
        with open(logfile_path, "r") as f:
            lines = f.readlines()
            iter_lines = [line for line in lines if line.strip().startswith(("*", " ", "0:")) and "obj =" in line]
            print(f"\nIteraciones: {len(iter_lines)}")
            for line in iter_lines:
                print(line.strip())

        return self.results

    def display_model(self):
        # Mostrar resultados
        print("\nResultado óptimo:")
        for i in self.model.I:
            print(f"x{i} = {value(self.model.x[i]):.2f}")
        print(f"Z = {value(self.model.obj):.2f}")

    def imprimir_tabla(tabla, Z, nombres, bases):
        print("\nTabla:")
        encabezado = nombres + ["RHS"]
        print("Base\t" + "\t".join(encabezado))
        for i, fila in enumerate(tabla):
            base = nombres[bases[i]] if bases[i] < len(nombres) else f"B{bases[i]}"
            print(f"{base}\t" + "\t".join(f"{x:.2f}" for x in fila))
        print("Z\t" + "\t".join(f"{x:.2f}" for x in Z))

    def preparar_tabla(self):
        c = [5, 8, 3, 7, 6, 9, 4, 10, 2, 11]
        A = [
                [1, 2, 1, 1, 0, 0, 3, 1, 2, 1],
                [2, 1, 0, 2, 1, 1, 0, 3, 1, 2],  
                [1, 1, 2, 0, 2, 1, 1, 0, 3, 1],  
                [0, 2, 1, 1, 1, 0, 2, 1, 1, 1],  
                [2, 0, 1, 1, 1, 2, 1, 1, 0, 2],  
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  
                [0, 1, 2, 1, 0, 1, 2, 1, 1, 0],  
                [1, 0, 1, 2, 1, 0, 1, 2, 1, 1]   
            ]
        b = [50, 60, 55, 40, 45, 70, 65, 50]
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

    def construir_Z(self,tabla, bases, artificiales):
        Z = np.zeros(tabla.shape[1])
        for i, b in enumerate(bases):
            if b in artificiales:
                Z -= tabla[i]
        return Z

    def pivotear(self,tabla, Z, fila, col, bases):
        tabla[fila] /= tabla[fila, col]
        for i in range(len(tabla)):
            if i != fila:
                tabla[i] -= tabla[i, col] * tabla[fila]
        Z -= Z[col] * tabla[fila]
        bases[fila] = col

    def simplex_2(self,tabla, Z, bases, nombres, fase=1):
        iteraciones = 0
        while True:
            if all(Z[:-1] >= -1e-8):
                break
            col = np.argmin(Z[:-1])
            if all(tabla[:, col] <= 0):
                raise ValueError("Problema no acotado.")
            ratios = [fila[-1] / fila[col] if fila[col] > 0 else np.inf for fila in tabla]
            fila = np.argmin(ratios)
            self.pivotear(tabla, Z, fila, col, bases)
            iteraciones += 1
        return tabla, Z, bases, iteraciones

    def eliminar_artificiales(self,tabla, Z, nombres, artificiales):
        tabla = np.delete(tabla, artificiales, axis=1)
        Z = np.delete(Z, artificiales)
        for i in sorted(artificiales, reverse=True):
            nombres.pop(i)
        return tabla, Z, nombres

    def simplex_dos_fases(self):
        c, tabla, bases, artificiales, nombres = self.preparar_tabla()

        # Fase I
        Z = self.construir_Z(tabla, bases, artificiales)
        tabla, Z, bases, iteraciones_f1 = self.simplex_2(tabla, Z, bases, nombres, fase=1)

        if abs(Z[-1]) > 1e-5:
            raise ValueError("Problema infactible.")

        # Eliminar artificiales
        tabla, Z, nombres = self.eliminar_artificiales(tabla, Z, nombres, artificiales)

        # Fase II
        c_ext = [-ci for ci in c] + [0]*(len(tabla[0])-len(c)-1)
        Z = np.zeros(len(c_ext)+1)
        Z[:-1] = c_ext
        for i, b in enumerate(bases):
            Z -= Z[b] * tabla[i]

        tabla, Z, bases, iteraciones_f2 = self.simplex_2(tabla, Z, bases, nombres, fase=2)

        solucion = np.zeros(len(c))
        for i, b in enumerate(bases):
            if b < len(c):
                solucion[b] = tabla[i, -1]

        valor_optimo = sum(c[i] * solucion[i] for i in range(len(c)))

        total_iteraciones = iteraciones_f1 + iteraciones_f2

        return solucion, valor_optimo, total_iteraciones


if __name__ == "__main__":

    sp_model = ProblemaTres(n=10, m=8, solver_name='glpk')

    print("\nImplementación con Pyomo + GLPK...")
    sp_model.setup_model()
    
    start_pyomo = time.perf_counter()
    results = sp_model.solve_model()
    end_pyomo = time.perf_counter()

    sp_model.display_model()
    print(f"Tiempo de ejecución Pyomo + GLPK: {end_pyomo - start_pyomo:.6f} segundos")

    print("Implementación implex Estándar")
    start_simplex = time.perf_counter()
    sp_model.solve_with_simplex()
    end_simplex = time.perf_counter()

    print(f"Tiempo de ejecución Simplex Estándar: {end_simplex - start_simplex:.6f} segundos")

    start = time.perf_counter()
    solucion, valor, iteraciones = sp_model.simplex_dos_fases()
    end = time.perf_counter()

    print("\nSolución óptima:", solucion)
    print("Valor mínimo de Z:", valor)
    print(f"Tiempo de ejecución: {end - start:.6f} segundos")
    print(f"Total de iteraciones: {iteraciones}")


    # Gráficas 
    n = 10  

    # Tiempos de ejecución en milisegundos
    tiempo_pyomo = (end_pyomo - start_pyomo) * 1000
    tiempo_estandar = (end_simplex - start_simplex) * 1000
    tiempo_dual = (end - start) * 1000


    # Número de iteraciones
    iter_pyomo = 2
    iter_estandar = 5
    iter_dual = 5

    plt.figure(figsize=(8, 4))
    plt.bar(
        ["Pyomo + GLPK", "Simplex Estándar", "Simplex Dual Phase"],
        [tiempo_pyomo, tiempo_estandar, tiempo_dual],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    plt.ylabel("Tiempo de ejecución (ms)")
    plt.title(f"Comparación de tiempo de ejecución (n = {n})")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Gráfica de número de iteraciones
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.bar(
        ["Pyomo + GLPK", "Simplex Estándar", "Simplex Dual Phase"],
        [iter_pyomo, iter_estandar, iter_dual],
        color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    plt.ylabel("Número de iteraciones")
    plt.title(f"Comparación de iteraciones (n = {n})")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()




