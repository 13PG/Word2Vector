# 文本聚类

## 这里展示了三种文本聚类的方式：
* TFIDF_cluster.py 采用传统的离散式文本表示方法:TF-IDF编码<br>
* word_cluster.py 采用传统的分布式:skip-Gram,word2doc,word2word等编码<br>
* tenxun_cluster.py  采用腾讯提供的开源文本向量数据，代表了大模型的文本编码处理方式<br>

## 其余文件夹的作用分别是：
* data: 存放TFIDF_cluster.py所需要的停用词数据，实验数据集以及word_cluster.py的实验数据<br>
* train_model:  存放word_cluster.py所需的配套模型【尚未训练】<br>

## 效果说明以及解释
* 离散式表示:one-hot编码,词袋编码,TF-IDF（词频-逆文档频率） <br>
* 分布式表示：word2vec,共现矩阵，text2vec等 <br>
#
随着因为离散式编码肯定是无法表示出词义的，并且分布假说明确了词义是由上下文决定的，因此要想表示出词义就必须得采用分布式的文本表示方法。但是随着大模型的发展，也涌现出了更多新的文本编码方式,比如说bert采用Transformer的编码器(Encoder)结构作为特征提取器,并使用与之配套的MLM训练方法,实现输入序列文本的双向编码。（*别看我，我也不知道这是个啥玩意*）<br>
**其实无论是文本聚类也好，文本分类也好，效果好不好的前提都是如何实现文本表示/文本编码**因此至少在这里三个模型的效果应该是：<br>
tenxun_cluster.py > word_cluster.py > TFIDF_cluster.py


