import random
import numpy as np
import os

node2vec = {}
# dataset_name = "zhihu"
f = open('temp/embed.txt', 'rb')
for i, j in enumerate(f):
    if j.decode() != '\n':
        node2vec[i] = list(map(float, j.strip().decode().split(' ')))
# f1 = open(os.path.join('temp/test_graph.txt'), 'rb')
# edges = [list(map(int, i.strip().decode().split('\t'))) for i in f1]
# nodes = list(set([i for j in edges for i in j]))
# a = 0
# b = 0
# for i, j in edges:
#     if i in node2vec.keys() and j in node2vec.keys():
#         dot1 = np.dot(node2vec[i], node2vec[j])
#         random_node = random.sample(nodes, 1)[0]
#         while random_node == j or random_node not in node2vec.keys():
#             random_node = random.sample(nodes, 1)[0]
#         dot2 = np.dot(node2vec[i], node2vec[random_node])
#         if dot1 > dot2:
#             a += 1
#         elif dot1 == dot2:
#             a += 0.5
#         b += 1
# print("Auc value:", float(a) / b)
# import random
# random.seed(20)
#
# idx_train = random.sample(range(3327), 1664)
# print(len(idx_train))
# idx_test = []
# for i in range(3327):
#     if i not in idx_train:
#         idx_test.append(i)
#
# with open('idx_train_r.txt', 'w') as fn:
#     for i in idx_train:
#         fn.write(str(i)+' ')
#
# with open('idx_test_r.txt', 'w') as fn:
#     for i in idx_test:
#         fn.write(str(i)+' ')
# print(len(idx_train)+len(idx_test))

train_value = []
test_value = []
label_value = []
with open('../datasets/citeseer/idx_train_r.txt', 'r') as fn:
# with open('idx_train_r.txt', 'r') as fn:
    x = fn.readline()
    x = x.strip().split()
    x = [int(i) for i in x]
    for i in x:
        if i in node2vec.keys():
            train_value.append(i)
train_value = np.array(train_value)
# print(train_value)
with open('../datasets/citeseer/idx_test_r.txt', 'r') as fn:
# with open('idx_test_r.txt', 'r') as fn:
    x = fn.readline()
    x = x.strip().split()
    # print(x)
    x = [int(i) for i in x]
    for i in x:
        # print('1111')
        if i in node2vec.keys():
            # print('2222')
            test_value.append(i)
test_value = np.array(test_value)
# print(test_value)
with open('../datasets/citeseer/group.txt', 'r') as fn:
    a = fn.readline()
    while a:
        a = a.strip().split()
        a = int(a[0])
        label_value.append(a)
        a = fn.readline()
label_value = np.array(label_value)

from sklearn import linear_model
model_LinearRegression = linear_model.LogisticRegression()
dim = len(node2vec[0])
model_LinearRegression.fit(np.array([node2vec[i] for i in train_value]).reshape(-1,dim),
                           np.array([label_value[i] for i in train_value]).reshape(-1,1))
score1 = model_LinearRegression.score(np.array([node2vec[i] for i in test_value]).reshape(-1,dim),
                                     np.array([label_value[i] for i in test_value]).reshape(-1,1))
print('precision:',score1)

model_LinearRegression2 = linear_model.LogisticRegression()
model_LinearRegression2.fit(np.array([node2vec[i] for i in test_value]).reshape(-1, dim),
                            np.array([label_value[i] for i in test_value]).reshape(-1, 1))
score2 = model_LinearRegression2.score(np.array([node2vec[i] for i in train_value]).reshape(-1, dim),
                                      np.array([label_value[i] for i in train_value]).reshape(-1, 1))
print('precision2:', score2)
print('ave_precision:',(score1+score2)/2)




