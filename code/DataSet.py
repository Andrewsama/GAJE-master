import sys

import config

import numpy as np
from tensorflow.contrib import learn
from negativeSample import InitNegTable
import random


class dataSet:
    def __init__(self, text_path, graph_path):

        text_file, graph_file = self.load(text_path, graph_path)

        self.edges = self.load_edges(graph_file)

        self.text, self.num_vocab, self.num_nodes = self.load_text(text_file)

        self.negative_table = InitNegTable(self.edges)

    def load(self, text_path, graph_path):
        text_file = open(text_path, 'rb').readlines()
        for a in range(0, len(text_file)):
            text_file[a] = str(text_file[a])
        graph_file = open(graph_path, 'rb').readlines()

        return text_file, graph_file

    def load_edges(self, graph_file):
        edges = []
        for i in graph_file:
            edges.append(list(map(int, i.strip().decode().split())))

        print("Total load %d edges." % len(edges))

        return edges

    def load_text(self, text_file):
        vocab = learn.preprocessing.VocabularyProcessor(config.MAX_LEN)#创建300维的词汇表
        text = np.array(list(vocab.fit_transform(text_file)))#text_file对应词汇表的矩阵
        num_vocab = len(vocab.vocabulary_)
        num_nodes = len(text)#词的数量，节点数
        # print('词汇表第一行:',text[2])
        # print('词数：',num_vocab)
        # print('节点数:',num_nodes)

        return text, num_vocab, num_nodes

    def negative_sample(self, edges):
        node1, node2 = zip(*edges)
        sample_edges = []
        func = lambda: self.negative_table[random.randint(0, config.neg_table_size - 1)]
        for i in range(len(edges)):
            neg_node = func()
            while node1[i] == neg_node or node2[i] == neg_node:
                neg_node = func()
            sample_edges.append([node1[i], node2[i], neg_node])

        return sample_edges

    def generate_batches(self, mode=None):

        num_batch = len(self.edges) // config.batch_size
        edges = self.edges
        # if mode == 'add':
        #     num_batch += 1
        #     edges.extend(edges[:(config.batch_size - len(self.edges) // config.batch_size)])
        if mode != 'add':
            random.shuffle(edges)
        sample_edges = edges[:num_batch * config.batch_size]
        sample_edges = self.negative_sample(sample_edges)

        batches = []
        for i in range(num_batch):
            batches.append(sample_edges[i * config.batch_size:(i + 1) * config.batch_size])
        # print sample_edges[0]
        return batches
