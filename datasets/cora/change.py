temp = []
# with open('group.txt','r') as fn:
#     a = fn.readline()
#     while a:
#         temp.append(a)
#         a = fn.readline()
# with open('groups.txt', 'w') as fn:
#     for i in range(len(temp)):
#         fn.write(str(i)+' '+temp[i])

# with open('cora_features.txt', 'r') as fn:
#     a = fn.readline()
#     while a:
#         temp.append(a)
#         a = fn.readline()
# print('节点数:',len(temp))
# with open('features.txt', 'w') as fn:
#     for i in range(len(temp)):
#         fn.write(str(i)+' '+temp[i])

# f = open('graph.txt', 'rb')
# edges = [i.strip().split() for i in f]
# with open('edge.txt', 'w') as fn:
#     for i in range(len(edges)):
#         fn.write(str(int(edges[i][0]))+' '+str(int(edges[i][1]))+'\n')

# from tensorflow.contrib import learn
# import numpy as np
# text_file = open('data.txt', 'rb').readlines()
# for a in range(0, len(text_file)):
#     text_file[a] = str(text_file[a])
# vocab = learn.preprocessing.VocabularyProcessor(300)#创建300维的词汇表
# text = np.array(list(vocab.fit_transform(text_file)))#text_file对应词汇表的矩阵
# num_vocab = len(vocab.vocabulary_)
# num_nodes = len(text)#词的数量，节点数
# print(text)
