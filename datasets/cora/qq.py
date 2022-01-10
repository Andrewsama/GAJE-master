import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_data(dataset_str): # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    return adj, features, labels, idx_train, idx_val, idx_test
adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')
features = features.tocoo()
# print(features.shape)
flag = 0
count = 0
'''data.txt'''
with open('data.txt', 'w') as ff:
    for i in features.row:
        if flag == i:
            ff.write(str(features.col[count]))
            ff.write(' ')
        else:
            while True:
                flag += 1
                ff.write('\n')
                if flag == i:
                    ff.write(str(features.col[count]))
                    ff.write(' ')
                    break
                else:
                    ff.write(str(299))
        count += 1
'''graph.txt'''
adj = adj.tocoo()
with open('graph.txt', 'w') as fn:
    for i,j in zip(adj.row, adj.col):
        fn.write(str(i)+' '+str(j)+'\n')
'''group.txt'''
print(labels.shape)
lab = np.argmax(labels,axis=1)
with open('group.txt', 'w') as fn:
    for i in lab:
        fn.write(str(i)+'\n')

'''train_test'''
with open('idx_train.txt', 'w') as fn:
    for i in idx_train:
        fn.write(str(i)+' ')
'''test'''
with open('idx_test.txt', 'w') as fn:
    for i in idx_test:
        fn.write(str(i)+' ')

# print(features.max)

