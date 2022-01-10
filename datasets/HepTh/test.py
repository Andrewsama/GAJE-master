import numpy as np
word_set = []    #单词表
word_matrix = [] #单词矩阵
with open('data.txt', 'r') as fn:
    a = fn.readline()
    while a:
        a = list(a.strip().split())
        word_set.extend(a)
        word_matrix.append(a)
        a = fn.readline()
word_set = list(set(word_set))
num_word = len(word_set)    #返回单词表词数
num_node = len(word_matrix) #返回节点数
print('节点数:',num_node)
print('单词数:',num_word)
word_index = {}
index_word = {}
for i in range(len(word_set)):
    word_index[word_set[i]] = i  # 返回词对应下标
    index_word[i] = word_set[i]  # 返回下标对应词
feature = np.zeros([num_node, num_word])
for i in range(len(word_matrix)):
    for j in range(len(word_matrix[i])):
        feature[i,word_index[word_matrix[i][j]]] +=1
with open('HepTh_features.txt', 'w') as fn:
    for i in range(num_node):
        fn.write(str(int(feature[i,0])))
        for j in range(1,num_word):
            fn.write(' '+str(int(feature[i,j])))
        fn.write('\n')


