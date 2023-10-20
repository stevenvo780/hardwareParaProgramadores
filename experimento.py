# Este script muestra cómo y dónde se almacena el código y los datos en el ordenador
# Importar las bibliotecas necesarias estas existen en la ROM
import array
import numpy as np
import cupy as cp
import timeit
import json

# Abrir el archivo JSON en modo lectura que existe en ROM
with open('matrices.json', 'r') as f:
    # Cargar el contenido del archivo como un objeto de Python
    data = json.load(f)

#Lo anterior es lo que mas latencia tendria en un programa al cargar los archivos que es cuando abre pues tarda en pasar los datos a la RAM

# Asignar las matrices a unas variables, aqui ya pasamos el dato de ROM a RAM
matriz_1 = data['matriz_1']
matriz_2 = data['matriz_2']

# Convertir las matrices a objetos de NumPy y CuPy
array_cpu = np.array(matriz_1) + np.array(matriz_2) # usa CPU
array_gpu = cp.array(matriz_1) + cp.array(matriz_2) # usa GPU

# Es importante tener presente que comunicar un dato que existe en CPU a GPU
# y vicebersa es lento por que este tiene que transitar de RAM por PCI a la 
# GPU, como asi mismo hace un uso intensivo de la ROM cargando o de

# pasamos un dato de GPU a CPU y viceversa
array_cpu = cp.asnumpy(array_gpu)
array_gpu = cp.asarray(array_cpu)

# Imprimir el tipo de datos de las matrices
print("Tipo de datos de la matriz en CPU:", type(array_cpu))
print("Tipo de datos de la matriz en GPU:", type(array_gpu))

# Definir una función para medir el tiempo de ejecución de una operación
def benchmark_processor(array, operation, n):
    start = timeit.default_timer()
    for i in range(n):
        operation(array, array)
    end = timeit.default_timer()
    return end - start

# Realizar una suma de matrices en la CPU usando NumPy
cpu_time = benchmark_processor(array_cpu, np.add, 999)

# Realizar una suma de matrices en la GPU usando CuPy
# Se necesita ejecutar una iteración piloto en la GPU primero para compilar y almacenar en caché la función kernel en la GPU
benchmark_processor(array_gpu, cp.add, 1)
gpu_time = benchmark_processor(array_gpu, cp.add, 999)

# Imprimir los tiempos de ejecución y el factor de aceleración de la GPU sobre la CPU
print("Tiempo de CPU (s):", cpu_time)
print("Tiempo de GPU (s):", gpu_time)
print("Aceleración de GPU sobre CPU:", cpu_time / gpu_time, "x")