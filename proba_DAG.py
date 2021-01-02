import re
import math
import collections
from base import TokenizerBase

class Tokenizer(TokenizerBase):
    """基于词典、DAG和最大概率路径的分词"""

    def __init__(self, words, total=None):
        self.words = words
        if total is None:
            total = sum(words.values())
        self.total = total
        self.logtotal = math.log(self.total)

    def find_word(self, sentence):
        # 计算最大概率路径
        DAG = self.create_DAG(sentence)
        size = len(sentence)
        dp = {}
        dp[size] = (0, 0)
        for i in reversed(range(size)):
            # 最大概率的词
            dp[i] = max(
                (self.word_logproba(sentence[i:j+1]) + dp[j+1][0], j) \
                for j in DAG[i]
            )

        # i = 0
        while i < size:
            j = dp[i][1] + 1
            # 逐词返回
            word = sentence[i:j]
            yield word
            i = j

    # def _create_DAG(self, sentence):
    #     # 构建等权重DAG词图，使用邻接表的方式存储图
    #     DAG = collections.defaultdict(list)
    #     size = len(sentence)
    #     for i in range(size):
    #         j = i
    #         word = sentence[i]
    #         while j < size and word in self.words:
    #             if self.words[word]:
    #                 DAG[i].append(j)
    #             j += 1
    #             word = sentence[i:j+1]
    #         if not DAG[i]:
    #             DAG[i].append(i)
    #     return DAG

    def create_DAG(self, sentence):
        # 构建等权重DAG词图，使用邻接表的方式存储图
        DAG = collections.defaultdict(list)
        size = len(sentence)
        i = 0
        while i < size:
            for j in range(i, size):
                word = sentence[i:j+1]
                if word in self.words and self.words[word]:
                    DAG[i].append(j)
            if not DAG[i]:
                DAG[i].append(i)
            i += 1
        return DAG

    def word_logproba(self, word):
        # 计算词的对数概率
        return math.log(self.words.get(word) or 1) - self.logtotal

if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    # words, total = dataset.load_chinese_words()
    tokenizer = Tokenizer(words, total)
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))

    # 测试分词的完整性
    text = dataset.load_human_history()
    words = tokenizer.cut(text)
    assert "".join(words) == text
