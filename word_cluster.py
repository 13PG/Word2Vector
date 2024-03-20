#!/usr/bin/env python3
# coding: utf-8
# File: word_cluster.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-25 
import numpy as np

class WordCluster:
    def __init__(self):
        # self.embedding_path = 'model/word2word_wordvec.bin'
        self.embedding_path = 'model/word2doc_wordvec.bin'          #效果较好
        # self.embedding_path = 'model/skipgram_wordvec.bin'
        # self.embedding_path = 'model/cbow_wordvec.bin'
        self.word_embedding_dict, self.word_dict, self.word_embeddings = self.load_model(self.embedding_path)
        self.similar_num = 10

    #加载词向量文件
    def load_model(self, embedding_path):
        print('loading models....')
        word_embedding_dict = {}
        word_embeddings = []
        word_dict = {}
        index = 0
        for line in open(embedding_path):
            line = line.strip().split('\t')
            word = line[0]
            word_embedding = np.array([float(item) for item in line[1].split(',') if item])
            word_embedding_dict[word] = word_embedding
            word_embeddings.append(word_embedding)
            word_dict[index] = word
            index += 1
        return word_embedding_dict, word_dict, np.array(word_embeddings)
    
    # 计算相似度【这里还暂时没搞懂，它用的什么方式算距离】
    def similarity_cosine(self, word):
        A = self.word_embedding_dict[word]
        B = (self.word_embeddings).T
        dot_num = np.dot(A, B)
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        cos = dot_num / denom
        sims = 0.5 + 0.5 * cos
        sim_dict = {self.word_dict[index]: sim for index, sim in enumerate(sims.tolist()) if word != self.word_dict[index]}
        sim_words = sorted(sim_dict.items(), key=lambda asd: asd[1], reverse=True)[:self.similar_num]
        return sim_words
    
    #获取相似词语
    def get_similar_words(self, word):
        if word in self.word_embedding_dict:
            return self.similarity_cosine(word)
        else:
            return []

def test():
    vec = WordCluster()
    while 1:
        word = input('enter an word to search:').strip()
        simi_words = vec.get_similar_words(word)
        for word in simi_words:
            print(word)

test()

'''
这个项目实现文本聚类的原理：
最后保存的模型其实就是把训练集里的每个分词都用向量表示出来了，
然后你自己去输训练集里有的词,他才会根据聚类去算距离。
不然你输入一个不存在的词,它是不会有联想功能的
所以对你的的输入数据集要求比较高,你要想功能越强大,你要喂的数据就越多

因为TF-IDF肯定是无法表示出词义的，但是分布假说明确了词义是由上下文决定的，要想表示出词义就必须得采用分布式的文本表示方法
我们这里采用的word2doc模型是用的共现矩阵，他是属于分布式的文本表示，这是一个改良点。
'''
