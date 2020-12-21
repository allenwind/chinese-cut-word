# chinese-cut-word

中文分词有两大难度：
- 歧义
- 分词规范不统一
- 未登录词，即新词

中文分词大致可以分为：词典匹配和逐字标注（4tags:SBME和6tags:SBMIEO）两大类方法。

## 词典匹配的分词方法

基于词典匹配的分词方法：
- 完全切分匹配（扫描句子，只要组成词就切分）
- 正向最长匹配
- 逆向最长匹配
- 双向最长匹配（正向最长匹配+逆向最长匹配，并**选择词数更少**的结果）

基于词典匹配和概率统计的分词方法：
- 基于DAG的最大概率路径组合
- 基于语言模型的最大概率路径组合
- 基于信息论的方案

以上这些方法涉及词的查找、匹配和路径计算，可以配合数据结构Trie树（双数组实现效率更高）、哈希表和AC自动机（Aho and Corasick）优化。其中Trie树复用词的公共前缀，节省内存。以上的基本方法可以进一步派生分词算法或策略，如期望句子切分粒度大一点，那么策略就是分词数最少。





## 逐字标注的分词方法

基于逐字标注的分词方法：
- HMM
- CRF
- 逐位置分类模型
- 深度模型+CRF

## 构建词库

词典需要通过新词发现的方案构造，新词发现与分词可以看做是互相迭代的同类算法，构建词库的方法：
- 基于信息熵的方案


## 对比

词典匹配和逐字标注对比：
- 基于逐字标注的方法对歧义词和未登录词的识别比单纯词典匹配的方法更好。


## 应用


[word-char-hybrid-embedding](https://github.com/allenwind/word-char-hybrid-embedding)


## 参考

[1] Combining Classifiers for Chinese Word Segmentation
[2] Chinese word segmentation as character tagging
[3] eural Architectures for Named Entity Recognition
