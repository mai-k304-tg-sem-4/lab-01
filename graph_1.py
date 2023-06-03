import numpy as np
import sys
import pandas as pd

class Graph:
    def __init__(self, file_path, file_type):
        # Конструктор класса Graph
        # Инициализирует объект графа с указанным путем к файлу и типом файла
        self.file_path = file_path
        self.file_type = file_type
        self.graph = self.load_graph()

    def load_graph(self):
        # Метод загрузки графа в зависимости от типа файла
        if self.file_type == "-e":
            return self.load_list_of_edges()  # Загрузка графа из списка ребер
        elif self.file_type == "-m":
            return self.load_adjacency_matrix()  # Загрузка графа из матрицы смежности
        elif self.file_type == "-l":
            return self.load_adjacency_list()  # Загрузка графа из списка смежности
        else:
            raise ValueError("Invalid file type")  # Вызов исключения, если тип файла некорректен

    def load_adjacency_list(self):
        # Метод загрузки графа из файла в формате списка смежности
        with open(self.file_path, 'r') as file:
            lines = file.readlines()  # Чтение всех строк из файла
        num_vertices = len(lines)  # Количество вершин в графе равно количеству строк
        matrix = np.zeros((num_vertices, num_vertices))  # Создание матрицы с нулевыми значениями
        matrix[:] = np.inf  # Установка всех элементов матрицы в бесконечность
        for i, line in enumerate(lines):
            neighbors = line.strip().split()  # Разделение строки на соседей вершины
            for neighbor in neighbors:
                matrix[i, int(neighbor) - 1] = 1  # Установка значения 1 для соответствующих соседних вершин
        return matrix  # Возвращение матрицы смежности

    def load_adjacency_matrix(self):
        # Метод загрузки графа из файла в формате матрицы смежности
        with open(self.file_path, 'r') as file:
            lines = file.readlines()  # Чтение всех строк из файла
        num_vertices = len(lines)  # Количество вершин в графе равно количеству строк
        matrix = np.zeros((num_vertices, num_vertices))  # Создание матрицы с нулевыми значениями
        matrix[:] = np.inf  # Установка всех элементов матрицы в бесконечность
        for i, line in enumerate(lines):
            row = line.strip().split()  # Разделение строки на элементы строки
            for j, value in enumerate(row):
                matrix[i, j] = int(value) if int(value) != 0 else np.inf  # Заполнение матрицы значениями из файла
        return matrix  # Возвращение матрицы смежности

    def adjacency_matrix(self):
        return self.graph  # Возвращение матрицы смежности графа

    def load_list_of_edges(self):
        # Метод загрузки графа из файла в формате списка ребер
        with open(self.file_path, 'r') as file:
            lines = file.readlines()  # Чтение всех строк из файла
        num_vertices = 0  # Инициализация количества вершин
        edges = []  # Инициализация списка ребер
        for line in lines:
            values = line.strip().split()  # Разделение строки на значения
            if len(values) == 2:
                vertex1, vertex2 = values  # Получение вершин ребра
                weight = 1  # Установка веса ребра по умолчанию
            else:
                vertex1, vertex2, weight = values  # Получение вершин и веса ребра
            edges.append((vertex1, vertex2, int(weight)))  # Добавление ребра в список ребер
            num_vertices = max(num_vertices, int(vertex1), int(vertex2))  # Обновление количества вершин
        matrix = np.zeros((num_vertices, num_vertices))  # Создание матрицы с нулевыми значениями
        matrix[:] = np.inf  # Установка всех элементов матрицы в бесконечность
        for edge in edges:
            vertex1, vertex2, weight = edge  # Получение вершин и веса ребра
            matrix[int(vertex1) - 1, int(vertex2) - 1] = weight  # Заполнение матрицы значениями из списка ребер
            matrix[int(vertex2) - 1, int(vertex1) - 1] = weight  # Заполнение матрицы значениями из списка ребер (для неориентированного графа)
        return matrix  # Возвращение матрицы смежности

    def list_of_edges(self, v):
        # Метод получения списка ребер для заданной вершины
        edges = []  # Инициализация списка ребер
        for i in range(self.graph.shape[0]):
            if self.graph[v - 1, i] != np.inf:
                edges.append((v, i + 1, self.graph[v - 1, i]))  # Добавление ребра в список ребер
        return edges  # Возвращение списка ребер

    def is_directed(self):
        return self.graph.transpose() == self.graph  # Проверка, является ли граф ориентированным

    def degree_vector(self):
        # Метод получения вектора степеней вершин
        if self.is_directed().any():
            degrees = np.sum(self.graph != np.inf, axis=1)  # Сумма значений по строкам для ориентированного графа
        else:
            degrees = np.sum(self.graph != np.inf, axis=1) - 1  # Сумма значений по строкам для неориентированного графа
        return degrees  # Возвращение вектора степеней вершин

    def distance_matrix(self):
        # Метод получения матрицы расстояний
        num_vertices = self.graph.shape[0]  # Количество вершин
        dist_matrix = np.copy(self.graph)  # Копирование матрицы смежности
        dist_matrix[dist_matrix == np.inf] = np.nan  # Замена бесконечных значений на NaN
        np.fill_diagonal(dist_matrix, 0)  # Заполнение диагонали нулями
        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if np.isnan(dist_matrix[i, k]) or np.isnan(dist_matrix[k, j]):
                        continue  # Пропуск некорректных значений
                    if np.isnan(dist_matrix[i, j]) or dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]  # Обновление значений матрицы расстояний
        dist_matrix[np.isnan(dist_matrix)] = np.inf  # Замена NaN на бесконечность
        return dist_matrix  # Возвращение матрицы расстояний

    def diameter(self):
        # Метод получения диаметра графа
        dist_matrix = self.distance_matrix()  # Получение матрицы расстояний
        diameter = np.max(dist_matrix[dist_matrix != np.inf])  # Нахождение максимального расстояния
        return diameter  # Возвращение диаметра

    def radius(self):
        # Метод получения радиуса графа
        dist_matrix = self.distance_matrix()  # Получение матрицы расстояний
        radius = np.min(np.max(dist_matrix, axis=1))  # Нахождение минимального максимального расстояния
        return radius  # Возвращение радиуса

    def central_vertices(self):
        # Метод получения центральных вершин графа
        dist_matrix = self.distance_matrix()  # Получение матрицы расстояний
        radius = self.radius()  # Получение радиуса
        central_vertices = np.where(np.max(dist_matrix, axis=1) == radius)[0] + 1  # Нахождение центральных вершин
        return central_vertices  # Возвращение центральных вершин

    def peripheral_vertices(self):
        # Метод получения периферийных вершин графа
        dist_matrix = self.distance_matrix()  # Получение матрицы расстояний
        diameter = self.diameter()  # Получение диаметра
        peripheral_vertices = np.where(np.max(dist_matrix, axis=1) == diameter)[0] + 1  # Нахождение периферийных вершин
        return peripheral_vertices  # Возвращение периферийных вершин

    def eccentricity(self):
        # Метод получения эксцентриситетов вершин графа
        distances = self.distance_matrix()  # Получение матрицы расстояний
        eccentricities = np.max(distances, axis=1)  # Нахождение максимального расстояния для каждой вершины
        return eccentricities  # Возвращение эксцентриситетов вершин

print("Введите ключ параметра:")
print("-e: list_of edges, \n-m: matrix, \n-l: list_of_adjacency")

key = input() 
if key not in ['-m', '-e', '-l']:
    print('Неверный тип ключа!')
    sys.exit()
print("Введите название файла (в текущем каталоге):")
file = input()
print('\n')

g = Graph(file, key)

adj_matrix = g.adjacency_matrix()
np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=8, suppress=True)

degree_vec = g.degree_vector()
print("deg =", degree_vec.astype(int), '\n')

dist_matrix = g.distance_matrix()
print("Distances:")
np.set_printoptions(threshold=dist_matrix.shape[0] // 2)
print(dist_matrix.astype(int), '\n')

adj_matrix = g.adjacency_matrix()
eccentricities = g.eccentricity()
print("Eccentricity:")
print(eccentricities.astype(int), '\n')

diameter = g.diameter()
print("D =", int(diameter))

radius = g.radius()
print("R =", int(radius))

central_vertices = g.central_vertices()
print("Z =", central_vertices.astype(int))

peripheral_vertices = g.peripheral_vertices()
print("P =", peripheral_vertices.astype(int))

# Запись результатов в файл
with open("output.txt", 'w') as file:
    file.write("Adjacency matrix:\n")
    for row in adj_matrix:
        row_str = ' '.join(map(str, row))
        file.write(row_str + '\n')
    file.write('\n')

    file.write("deg =")
    np.savetxt(file, degree_vec.astype(int), fmt='%d', delimiter=' ')
    file.write("\n\n")

    file.write("Distances:\n")
    np.savetxt(file, dist_matrix.astype(int), fmt='%d', delimiter=' ')
    file.write("\n")

    file.write("Eccentricity:\n")
    np.savetxt(file, eccentricities.astype(int).astype(str), fmt='%s', delimiter=' ')
    file.write("\n")

    file.write("D = ")
    file.write(str(int(diameter)))
    file.write("\n")

    file.write("R = ")
    file.write(str(int(radius)))
    file.write("\n")

    file.write("Z = ")
    np.savetxt(file, central_vertices.astype(int), fmt='%d', delimiter=' ')
    file.write("\n")

    file.write("P = ")
    np.savetxt(file, peripheral_vertices.astype(int), fmt='%d', delimiter=' ')

print("\nРезультаты записаны в файл output.txt")