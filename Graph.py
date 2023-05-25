import networkx as nx
import numpy as np

class Graph(object):
    def __init__(self, type, file_name):
        if (type == 'm'):
            self.matrix = open(file_name, "r").read()
        self.matrix = '[' + self.matrix.replace(' ', ',').replace(',\n', ';')[:-1] + ']'
        self.matrix = np.matrix(self.matrix, dtype=np.int32)
        self.graph = nx.DiGraph(self.matrix)
        #if (type == 'e'):

    def weight(self, vi, vj):
        return self.matrix[vi].item(vj)
    def is_edge(self, vi, vj):
        if (self.matrix[vi].item(vj) != 0):
            return True
        else:
            return False
    def adjacency_matrix(self):
        return self.matrix
    def adjacency_list(self, v):
        return_array = np.array([])
        for i in range(0, self.matrix[v].size):
            if (self.matrix[v].item(i) != 0):
                return_array = np.append(return_array, i)
        for i in range(0, self.matrix[v].size):
            if (self.matrix[i].item(v) != 0):
                return_array = np.append(return_array, i)
        return return_array

    def list_of_edges(self, v=None):
        return_array = np.array([])
        if (v == None):
            for i in range(0, self.matrix[0].size):
                for j in range(0, self.matrix[0].size):
                    if (self.matrix[i].item(j) != 0):
                        return_array = np.append(return_array, [i, j])
        else:
            for i in range(0, self.matrix[v].size):
                if (self.matrix[v].item(i) != 0):
                    return_array = np.append(return_array, i)
        return return_array

    def is_directed(self):
        for i in range(0, self.matrix[0].size):
            for j in range(0, self.matrix[0].size):
                if (self.matrix[i].item(j) > 1):
                        return True
        return False