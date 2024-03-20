# -*- coding: utf-8 -*-

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from hanlp_restful import HanLPClient
import os
import numpy as np
os.environ['HF_ENDPOINT']='https://hf-mirror.com'

class KmeansClustering():
    def __init__(self, stopwords_path=None):
        self.stopwords = self.load_stopwords(stopwords_path)
        self.vectorizer = CountVectorizer()
        self.transformer = TfidfTransformer()

    def load_stopwords(self, stopwords=None):
        """
        加载停用词
        :param stopwords:
        :return:
        """
        if stopwords:
            with open(stopwords, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f]
        else:
            return []

    def preprocess_data(self, corpus_path):
        """
        文本预处理，每行一个文本
        :param corpus_path:
        :return:
        """
        corpus = []
        HanLP = HanLPClient('https://www.hanlp.com/api', auth="NDQyMEBiYnMuaGFubHAuY29tOjYyR1hwQjgwR01MUktkUzc=", language='zh') # auth不填则匿名，zh中文，mul多语种
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = ' '.join([word for word in jieba.lcut(line.strip()) if word not in self.stopwords])
                corpus.append(s)
        return corpus

    def get_text_tfidf_matrix(self, corpus):
        """
        获取tfidf矩阵
        :param corpus:  
        :return:
        """
        tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(corpus))
        ##这里其实会把所有词汇都提取出来，作为语料库/词袋

        # 获取词袋中所有词语
        words = self.vectorizer.get_feature_names_out()
        print("词袋里有{}".format(words))

        # 获取tfidf矩阵中权重
        weights = tfidf.toarray()
        # print("tfidf矩阵权重{}----{}".format(weights,np.mat(weights).shape))      #每一行表示对应的文本在tfidf下的向量表示
        return weights

    def kmeans(self, corpus_path, n_clusters=5):
        """
        KMeans文本聚类
        :param corpus_path: 语料路径（每行一篇）,文章id从0开始
        :param n_clusters: ：聚类类别数目
        :return: {cluster_id1:[text_id1, text_id2]}
        """
        corpus = self.preprocess_data(corpus_path)
        weights = self.get_text_tfidf_matrix(corpus)

        clf = KMeans(n_clusters=n_clusters)

        y = clf.fit_predict(weights)

        # 中心点
        centers = clf.cluster_centers_

        # 用来评估簇的个数是否合适,距离约小说明簇分得越好,选取临界点的簇的个数
        score = clf.inertia_    

        print("当前的中心点是{},得分是{}".format(centers,score))
        # 每个样本所属的簇
        result = {}
        for text_idx, label_idx in enumerate(y):
            if label_idx not in result:
                result[label_idx] = [text_idx]
            else:
                result[label_idx].append(text_idx)
        return result


if __name__ == '__main__':
    Kmeans = KmeansClustering(stopwords_path=r'C:\Users\Administrator\Desktop\text_clustering-master\data\stop_words.txt')          #指定停用词的路径
    result = Kmeans.kmeans(r'C:\Users\Administrator\Desktop\text_clustering-master\data\resume_data.txt', n_clusters=2)             #指定数据集【这个其实没有在训练模型，用的就是TFIDF,这里实际上是测试集】
    print(result)
    
'''
这个的聚类方法是根据已经存在的语料库:每一行代表一篇文章
预处理的时候去除停用词,提取每篇的词汇,采用TF-IDF的方式,生成对应每篇的向量表示
再sklearn的聚类包实现聚类
'''
