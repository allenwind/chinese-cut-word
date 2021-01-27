import collections
import numpy as np
from base import TokenizerBase

# 最短路分词、N-最短路分词

def _create_DAG(sentence, words):
    # 构建等权重DAG词图，使用邻接表的方式存储图
    DAG = collections.defaultdict(list)
    size = len(sentence)
    for i in range(size):
        j = i
        word = sentence[i]
        while j < size and word in words:
            if words[word]:
                DAG[i].append(j)
            j += 1
            word = sentence[i:j+1]
        if not DAG[i]:
            DAG[i].append(i)
    return DAG

def create_DAG(sentence, words):
    # 构建等权重DAG词图，使用邻接表的方式存储图
    DAG = collections.defaultdict(list)
    size = len(sentence)
    i = 0
    while i < size:
        for j in range(i, size):
            word = sentence[i:j+1]
            if word in words and words[word]:
                DAG[i].append(j)
        if not DAG[i]:
            DAG[i].append(i)
        i += 1
    return DAG

def dijkstra(D, start=None, end=None):
    # https://zhuanlan.zhihu.com/p/129373740
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

if __name__ == "__main__":
    import dataset
    import pprint
    words, total = dataset.load_freq_words()
    text = "独立自主和平等互利的原则"
    DAG = create_DAG(text, words)
    pprint.pprint(DAG)

# defaultdict(<class 'list'>,
#             {0: [0, 1, 3],
#              1: [1],
#              2: [2, 3],
#              3: [3],
#              4: [4, 5],
#              5: [5, 6, 8],
#              6: [6],
#              7: [7, 8],
#              8: [8],
#              9: [9],
#              10: [10, 11],
#              11: [11]})

# defaultdict(<class 'list'>,
#             {0: [0, 1, 3],
#              1: [1],
#              2: [2, 3],
#              3: [3],
#              4: [4, 5],
#              5: [5, 6, 8],
#              6: [6],
#              7: [7, 8],
#              8: [8],
#              9: [9],
#              10: [10, 11],
#              11: [11]})
