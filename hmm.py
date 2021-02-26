import itertools
import random
import math
import re
import numpy as np
from collections import *
from base import TokenizerBase

# HMM的实现，支持参数训练

class HiddenMarkovChain:
    
    def __init__(self):
        # 标签集
        self.tags = sorted("BMES")
        self.tags2id = {i:j for j,i in enumerate(self.tags)}
        self.id2tags = {j:i for i,j in self.tags2id.items()}
        self.state_size = len(self.tags)
        self.reset()

    def reset(self):
        # 状态转移矩阵
        self.A = np.zeros((self.state_size, self.state_size))
        # 观察矩阵，稀疏形式
        self.B = defaultdict(Counter)
        self.built = False

    def fit(self, X, y):
        if self.built:
            self.reset()
        # 状态转移矩阵参数学习
        for labels in y:
            for label1, label2 in zip(labels[:-1], labels[1:]):
                id1 = self.tags2id[label1]
                id2 = self.tags2id[label2]
                self.A[id1][id2] += 1
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        # 观察矩阵参数学习
        for sentence, labels in zip(X, y):
            for char, label in zip(sentence, labels):
                self.B[label][char] += 1
        self.logtotal = {tag:math.log(sum(self.B[tag].values())) for tag in self.tags}
        self.built = True

    def predict(self, X):
        batch_scores = []
        for sentence in X:
            scores = np.zeros((len(sentence), self.state_size))
            for i, char in enumerate(sentence):
                for j, k in self.B.items():
                    scores[i][self.tags2id[j]] = math.log(k[char]+1) - self.logtotal[j]
            batch_scores.append(scores)
        return batch_scores

    def viterbi_decode(self, scores, trans, return_score=False):
        # 使用viterbi算法求最优路径
        # scores.shape = (seq_len, num_tags)
        # trans.shape = (num_tags, num_tags)
        dp = np.zeros_like(scores)
        backpointers = np.zeros_like(scores, dtype=np.int32)
        dp[0] = scores[0]
        for t in range(1, scores.shape[0]):
            v = np.expand_dims(dp[t-1], axis=1) + trans
            dp[t] = scores[t] + np.max(v, axis=0)
            backpointers[t] = np.argmax(v, axis=0)

        viterbi = [np.argmax(dp[-1])]
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()
        if return_score:
            viterbi_score = np.max(dp[-1])
            return viterbi, viterbi_score
        return viterbi

class HMMTokenizer(TokenizerBase, HiddenMarkovChain):

    def find_word(self, sentence):
        scores = self.predict([sentence])[0]
        log_trans = np.log(np.where(self.A==0, 0.0001, self.A))
        viterbi = self.viterbi_decode(scores, log_trans)
        tags = [self.id2tags[i] for i in viterbi]
        buf = ""
        for tag, char in zip(tags, sentence):
            # t is S or B
            if tag in ["S", "B"]:
                if buf:
                    yield buf
                buf = char
            # t is M or E
            else:
                buf += char
        if buf:
            yield buf

PATH_CTB6 = "dataset/ctb6_cws/{}.txt"
def load_cws_ctb6(file, shuffle=True, with_labels=False):
    assert file in ("train", "dev", "test", "all")
    if file == "all":
        X = []
        y = []
        for file in ("train", "dev", "test"):
            X1, y1 = load_cws_ctb6(file)
            X.extend(X1)
            y.extend(y1)
        return X, y

    file = PATH_CTB6.format(file)
    with open(file, "r") as fp:
        text = fp.read()
    sentences = text.splitlines()
    if shuffle:
        random.shuffle(sentences)
    sentences = [re.split("\s+", sentence) for sentence in sentences]
    sentences = [[w for w in sentence if w] for sentence in sentences]

    X = []
    y = []
    for sentence in sentences:
        X.append("".join(sentence))
        tags = []
        for word in sentence:
            if len(word) == 1:
                tags.append("S")
            else:
                tags.extend(["B"] + ["M"]*(len(word)-2) + ["E"])
        y.append(tags)
        assert len("".join(sentence)) == len(tags)

    if with_labels:
        labels = sorted("BMES")
        return X, y, labels
    return X, y

if __name__ == "__main__":
    import dataset

    tokenizer = HMMTokenizer()
    X, y = load_cws_ctb6("all")
    tokenizer.fit(X, y)

    for text in dataset.load_sentences():
        print(tokenizer.cut(text))
