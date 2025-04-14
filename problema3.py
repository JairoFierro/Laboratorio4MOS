import numpy as np
import sympy as sp

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pyomo.environ import *

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

        # Variables
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
        self.results = solver.solve(self.model)

        return self.results

    def display_model(self):
        # Mostrar resultados
        print("\nResultado óptimo:")
        for i in self.model.I:
            print(f"x{i} = {value(self.model.x[i]):.2f}")
        print(f"Z = {value(self.model.obj):.2f}")

if __name__ == "__main__":

    sp_model = ProblemaTres(n=10,m=8, solver_name='glpk')

    # Set up the model
    sp_model.setup_model()

    # Resolver el modelo
    results = sp_model.solve_model()

    display_results = sp_model.display_model()

    print("\n Resolver con implementación Simplex Estándar propia")
    sp_model.solve_with_simplex()


