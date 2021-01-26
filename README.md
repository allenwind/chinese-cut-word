# chinese-cut-word

中文分词（chinese word segment，cws）有三大难度：
- 歧义
- 未登录词，即新词（如地面、人名、专业术语）
- 分词规范不统一（如“上海浦东”，可以当做一个词，也可以分为“上海”和“浦东”）

中文分词大致可以分为：词典匹配和逐字标注（4tags:SBME和6tags:SBMIEO）两大类方法。

## 词典匹配的分词方法

文本的扫描方向：正向和逆向，词匹配策略：最长词优先和最短词优先。配合词典这两种方式有如下具体分词方法。

基于词典匹配的分词方法：
- 完全切分匹配（扫描句子，只要组成词就切分）
- 正向最长匹配
- 逆向最长匹配
- 正向最小匹配（汉语中通常是成词出现，因此较少用）
- 逆向最小匹配
- 双向最长匹配（正向最长匹配+逆向最长匹配，并使用最少切分策略即**选择词数更少**的结果）

这些方法本质上是查词典，为避免无效的扫描，可以添加约束，如词的最大长度，遇到停用词跳出循环等等。

基于词典匹配和概率统计的分词方法，在词典的基础上进一步利用词频信息：
- 基于DAG的最大概率路径组合
- 基于语言模型的最大概率路径组合
- 基于信息论的方案

以上这些方法涉及词的查找、匹配和路径计算，可以配合数据结构Trie树（双数组实现效率更高）、哈希表、使用邻接表的方式存储DAG和AC自动机（Aho and Corasick）优化。其中Trie树复用词的公共前缀，节省内存。以上的基本方法可以进一步派生分词算法或策略，如期望句子切分粒度大一点，那么策略就是分词数最少。


## 逐字标注的分词方法

基于逐字标注的分词方法：
- HMM
- CRF
- 逐位置分类模型
- 深度模型+CRF（CNN、RNN）

## 构建词库

词典需要通过新词发现的方案构造，新词发现与分词可以看做是互相迭代的同类算法，构建词库的方法：
- 基于信息熵的方案
- 通过语言模型分词
- 通过相关矩阵阈值切分


## 对比

词典匹配和逐字标注对比：
- 基于逐字标注的方法对歧义词和未登录词的识别比单纯词典匹配的方法更好。

利用好中文中句和段有天然的空格或换行特点切分为互不影响的块。

逐字标注方法对未登录词的召回比词典方法要好。


## 工程实践

现实场景：
- 中英混合
- 长文

中英混合文本的分词技巧，提供基类便于长文分词，见`base.py`。继承`TokenizerBase`类，分词的基类，继承该类并在find_word实现分词的核心算法。

涉及内部状态记录


## 应用


[word-char-hybrid-embedding](https://github.com/allenwind/word-char-hybrid-embedding)

以上分词算法的并行化方案可参看[count-in-parallel](https://github.com/allenwind/count-in-parallel)


## 参考

[1] Combining Classifiers for Chinese Word Segmentation
[2] Chinese word segmentation as character tagging
[3] eural Architectures for Named Entity Recognition
[4] https://pyahocorasick.readthedocs.io/en/latest/
[5] 《统计自然语言处理》
