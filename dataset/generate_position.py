import scipy.sparse as sp
import numpy as np
import networkx as nx

data_name = 'citeseer'
data_path = './'+data_name+'/ori_adj.npz'

ori_view = sp.load_npz(data_path)
g = nx.from_scipy_sparse_matrix(ori_view)

number_nodes = ori_view.shape[0]
choice_number = int(np.log2(number_nodes))

row, col, short_path = [], [], []
choice_nodes = np.random.choice(g.number_of_nodes(),choice_number,replace=False)
for i in range(number_nodes):
    for j in range(choice_number):
        try:
            distance=nx.shortest_path_length(g, source=i, target=choice_nodes[j])
            row.append(i)
            col.append(j)
            short_path.append(distance)
        except nx.NetworkXNoPath:
            pass

distance_matrix = sp.coo_matrix((short_path, (row, col)), shape=(number_nodes, choice_number))

sp.save_npz("./"+data_name+"/position.npz", distance_matrix)
