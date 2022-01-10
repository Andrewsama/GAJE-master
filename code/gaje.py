import numpy as np
import tensorflow as tf
import config

class Model:
    def __init__(self, vocab_size, num_nodes, rho):
        rho = rho.split(",")
        self.rho1 = float(rho[0])
        self.rho2 = float(rho[1])
        self.rho3 = float(rho[2])
        # '''hyperparameter'''
        with tf.name_scope('read_inputs') as scope:
            self.Text_a = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Ta')#一个batch数64*一条游走序列300个节点
            self.Text_b = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tb')
            self.Text_neg = tf.placeholder(tf.int32, [config.batch_size, config.MAX_LEN], name='Tneg')
            self.Node_a = tf.placeholder(tf.int32, [config.batch_size], name='n1')#64*1
            self.Node_b = tf.placeholder(tf.int32, [config.batch_size], name='n2')
            self.Node_neg = tf.placeholder(tf.int32, [config.batch_size], name='n3')

        with tf.name_scope('initialize_embedding') as scope:
            self.text_embed = tf.Variable(tf.truncated_normal([vocab_size, config.embed_size // 2], stddev=0.3))#vocab_size*100
            self.node_embed = tf.Variable(tf.truncated_normal([num_nodes, config.embed_size // 2], stddev=0.3))#num_nodes*100
            self.node_embed = tf.clip_by_norm(self.node_embed, clip_norm=1, axes=1)#对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式。

        with tf.name_scope('lookup_embeddings') as scope:
            #text_A里面值是必须要小于等于text_embed的最大维度减一的
            self.TA = tf.nn.embedding_lookup(self.text_embed, self.Text_a)#look_up：64*300*100维
            self.T_A = tf.expand_dims(self.TA, -1)#维度在最后增加一维.比如【2，3，4】变成【2，3，4，1】:64*300*100*1

            self.TB = tf.nn.embedding_lookup(self.text_embed, self.Text_b)
            self.T_B = tf.expand_dims(self.TB, -1)

            self.TNEG = tf.nn.embedding_lookup(self.text_embed, self.Text_neg)
            self.T_NEG = tf.expand_dims(self.TNEG, -1)

            self.N_A = tf.nn.embedding_lookup(self.node_embed, self.Node_a)#64*1*100
            self.N_B = tf.nn.embedding_lookup(self.node_embed, self.Node_b)
            self.N_NEG = tf.nn.embedding_lookup(self.node_embed, self.Node_neg)
        self.convA, self.convB, self.convNeg = self.conv()
        self.loss = self.compute_loss()

    def conv(self):
        W2 = tf.Variable(tf.truncated_normal([2, config.embed_size // 2, 1, 100], stddev=0.3))#卷积核：2*100。1个通道。100个卷积核
        #rand_matrix = tf.Variable(tf.truncated_normal([100, 100], stddev=0.3))#注意力矩阵：100*100

        convA = tf.nn.conv2d(self.T_A, W2, strides=[1, 1, 1, 1], padding='VALID')#输出为64*299*1*100
        convB = tf.nn.conv2d(self.T_B, W2, strides=[1, 1, 1, 1], padding='VALID')
        convNEG = tf.nn.conv2d(self.T_NEG, W2, strides=[1, 1, 1, 1], padding='VALID')

        hA = tf.tanh(tf.squeeze(convA))#删除卷积层输出的1那个维度 输出维度为64*299*100
        hB = tf.tanh(tf.squeeze(convB))
        hNEG = tf.tanh(tf.squeeze(convNEG))

        r1 = tf.matmul(hA, hB, adjoint_b=True)#输出为64*299*299维矩阵.adjoint_b: 如果为真, b则在进行乘法计算前进行共轭和转置。
        r3 = tf.matmul(hA, hNEG, adjoint_b=True)

        att1 = tf.expand_dims(tf.stack(r1), -1)#64*299*299*1
        att3 = tf.expand_dims(tf.stack(r3), -1)

        att1 = tf.tanh(att1)
        att3 = tf.tanh(att3)

        pooled_A = tf.reduce_mean(att1, 2)#64*299*299*1—->64*299*1.制定2这个维度降维。0这个维度代表列
        pooled_B = tf.reduce_mean(att1, 1)#64*299*1
        pooled_NEG = tf.reduce_mean(att3, 1)

        a_flat = tf.squeeze(pooled_A)#64*299
        b_flat = tf.squeeze(pooled_B)
        neg_flat = tf.squeeze(pooled_NEG)

        w_A = tf.nn.softmax(a_flat)
        w_B = tf.nn.softmax(b_flat)
        w_NEG = tf.nn.softmax(neg_flat)

        rep_A = tf.expand_dims(w_A, -1)#64*299*1
        rep_B = tf.expand_dims(w_B, -1)
        rep_NEG = tf.expand_dims(w_NEG, -1)

        hA = tf.transpose(hA, perm=[0, 2, 1])#64*100*299.制定维度转置，按021这样排列
        hB = tf.transpose(hB, perm=[0, 2, 1])
        hNEG = tf.transpose(hNEG, perm=[0, 2, 1])

        rep1 = tf.matmul(hA,rep_A)#64*100*1
        rep2 = tf.matmul(hB,rep_B)
        rep3 = tf.matmul(hNEG,rep_NEG)

        attA = tf.squeeze(rep1)#嵌入64*100    嵌入维度为100维
        attB = tf.squeeze(rep2)
        attNEG = tf.squeeze(rep3)

        return attA, attB, attNEG #

    def compute_loss(self):
        p1 = tf.reduce_sum(tf.multiply(self.convA, self.convB), 1)
        p1 = tf.log(tf.sigmoid(p1) + 0.001)

        p2 = tf.reduce_sum(tf.multiply(self.convA, self.convNeg), 1)
        p2 = tf.log(tf.sigmoid(-p2) + 0.001)

        p3 = tf.reduce_sum(tf.multiply(self.N_A, self.N_B), 1)
        p3 = tf.log(tf.sigmoid(p3) + 0.001)

        p4 = tf.reduce_sum(tf.multiply(self.N_A, self.N_NEG), 1)
        p4 = tf.log(tf.sigmoid(-p4) + 0.001)

        p5 = tf.reduce_sum(tf.multiply(self.convB, self.N_A), 1)
        p5 = tf.log(tf.sigmoid(p5) + 0.001)

        p6 = tf.reduce_sum(tf.multiply(self.convNeg, self.N_A), 1)
        p6 = tf.log(tf.sigmoid(-p6) + 0.001)

        p7 = tf.reduce_sum(tf.multiply(self.N_B, self.convA), 1)
        p7 = tf.log(tf.sigmoid(p7) + 0.001)

        p8 = tf.reduce_sum(tf.multiply(self.N_B, self.convNeg), 1)
        p8 = tf.log(tf.sigmoid(-p8) + 0.001)

        rho1 = self.rho1
        rho2 = self.rho2
        rho3 = self.rho3
        temp_loss = rho1 * (p1 + p2) + rho2 * (p3 + p4) + rho3 * (p5 + p6) + rho3 * (p7 + p8)
        loss = -tf.reduce_sum(temp_loss)
        return loss
