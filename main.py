import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from Graph import *


A = Graph('m', "task1/matrix_t1_007.txt")

print(A.weight(1, 2))
print(A.is_edge(3, 1))
print(A.adjacency_matrix())
print(A.adjacency_list(1))
print(A.list_of_edges())
print(A.list_of_edges(1))
print(A.is_directed())


nx.draw(A.graph, pos=nx.spring_layout(A.graph), with_labels=True, node_size=300, arrows=True)
plt.show()