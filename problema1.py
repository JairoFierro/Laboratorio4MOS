import numpy as np
import matplotlib.pyplot as plt


# Coeficientes de la función objetivo negados 
# debido a que es un problema de maximización
c = np.array([-3, -2, -5, 0, 0, 0])  # x1, x2, x3, s1, s2, s3 

# Matriz A de restricciones con variables de holgura
A = np.array([
    [1, 1, 1, 1, 0, 0],   # fila de s1
    [2, 1, 1, 0, 1, 0],   # fila de s2
    [1, 4, 2, 0, 0, 1]    # fila de s3
])

# Terminos independientes valores b 
b = np.array([[100], [150], [80]])

# Unir A y b 
tabla_simplex_inic = np.hstack((A, b))

# Añadir la fila de la función objetivo
fila_z = np.append(c, [0]) 
tabla_simplex_inic = np.vstack((tabla_simplex_inic, fila_z))

# Variaables de holgura
variables_basicas = ['s1', 's2', 's3']

# Variables no básicas
variables_no_basicas = ['x1', 'x2', 'x3']


# El segundo paso del método simplex estandar es verificar optimalidad

def verificar_optimalidad(tabla_simplex):

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

def seleccionar_variable_entrada(tabla_simplex):
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

def calcular_variable_salida(tabla_simplex, col_entrada):
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

def operacion_pivoteo(tabla_simplex, fila_salida, col_entrada, variables_basicas):
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
def simplex(tabla_simplex, variables_basicas):
    iteracion = 1
    while True:
        print("\nIteración " + str(iteracion))
        if verificar_optimalidad(tabla_simplex):
            print("\nSolución óptima.")
            break

        col_entrada = seleccionar_variable_entrada(tabla_simplex)
        if col_entrada is None:
            print("No hay variable válida de entrada.")
            break

        fila_salida = calcular_variable_salida(tabla_simplex, col_entrada)
        if fila_salida is None:
            print("El problema es ilimitado.")
            break

        tabla_simplex, variables_basicas = operacion_pivoteo(
            tabla_simplex, fila_salida, col_entrada, variables_basicas
        )
        iteracion += 1

    return tabla_simplex, variables_basicas


# Ejecutar el método Simplex
tabla_final, variables_finales = simplex(tabla_simplex_inic, variables_basicas)

# Mostrar resultado final
print("\nSolución:")
for i, var in enumerate(variables_finales):
    valor = tabla_final[i, -1]
    print(f"{var} = {valor:.2f}")

print(f"\nValor óptimo de Z: {tabla_final[-1, -1]:.2f}")


# Gráficas

x1 = np.linspace(0, 100, 400)

x2 = 0

# Despejamos x3 de las restricciones, asumiendo x2 = 0
r1 = 100 - x1           
r2 = 150 - 2 * x1      
r3 = (80 - x1) / 2   

# Calcular la región factible
x3_max = np.minimum.reduce([r1, r2, r3])
x3_max = np.maximum(x3_max, 0)

plt.figure(figsize=(8,6))
plt.plot(x1, r1, label=r'$x_1 + x_3 \leq 100$')
plt.plot(x1, r2, label=r'$2x_1 + x_3 \leq 150$')
plt.plot(x1, r3, label=r'$x_1 + 2x_3 \leq 80$')


plt.fill_between(x1, 0, x3_max, color='lightblue', alpha=0.4, label='Región factible')

plt.plot(73.33, 3.33, 'ro', label='Solución óptima')

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_3$')
plt.title('Región factible (con $x_2 = 0$)')
plt.grid(True)
plt.legend()
plt.show()

