import math
from collections import Counter
from collections import defaultdict
import numpy as np

from viterbi import viterbi_decode as _viterbi_decode
from base import TokenizerBase

import dataset
words, total = dataset.load_freq_words()

def build_hmm_model(words):
    hmm_model = defaultdict(Counter)
    for word, freq in words.items():
        if len(word) == 1:
            hmm_model["s"][word] += freq
        else:
            hmm_model["b"][word[0]] += freq
            hmm_model["e"][word[-1]] += freq
            for c in word[1:-1]:
                hmm_model["m"][c] += freq
    return hmm_model

hmm_model = build_hmm_model(words)
tags = "sbme"
tags2id = {i:j for j,i in enumerate(tags)}
log_total = {tag:math.log(sum(hmm_model[tag].values())) for tag in tags}

def trans_humanize(trans):
    # 把转移矩阵转换为人类可读形式
    tags = "SBME"
    htrans = {}
    for i in range(4):
        for j in range(4):
            if trans[i][j] != 0.0:
                h = tags[i] + tags[j]
                htrans[h] = trans[i][j]
    return htrans

def get_trans(T=1, log=True):
    # 转移矩阵一
    _trans1 = [[0.3, 0.7, 0.0, 0.0], 
               [0.0, 0.0, 0.3, 0.7], 
               [0.0, 0.0, 0.3, 0.7], 
               [0.3, 0.7, 0.0, 0.0]]

    # 转移矩阵二
    _trans2 = [[0.514, 0.486, 0.0, 0.0],
               [0.0, 0.0, 0.400, 0.600],
               [0.0, 0.0, 0.284, 0.716],
               [0.446, 0.554, 0.0, 0.0]]

    # 转移矩阵三
    _trans3 = [[0.300, 0.700, 0.0, 0.0],
               [0.0, 0.0, 0.400, 0.600],
               [0.0, 0.0, 0.284, 0.716],
               [0.446, 0.554, 0.0, 0.0]]

    name = "_trans" + str(T)
    _trans = locals()[name]

    _trans = np.array(_trans)
    _trans = np.where(_trans==0, 0.0001, _trans)
    if log:
        _trans = np.log(_trans)
    return _trans

_log_trans = get_trans(T=2)

def predict(sentence):
    scores = np.zeros((len(sentence), 4))
    for i, c in enumerate(sentence):
        for j, k in hmm_model.items():
            scores[i][tags2id[j]] = math.log(k[c]+1) - log_total[j]
    return scores

def hmm_tokenize(sentence):
    scores = predict(sentence)
    viterbi = _viterbi_decode(scores, _log_trans)
    words = [sentence[0]]
    for i in range(1, len(sentence)):
        if viterbi[i] in [0, 1]:
            words.append(sentence[i])
        else:
            words[-1] += sentence[i]
    return words

class HMMTokenizer(TokenizerBase):
    """基于HMM逐字标注的分词方法"""

    def __init__(self, words, trans, method="viterbi"):
        self.trans = trans
        assert method in ("viterbi", "greedy")
        self.method = method
        if self.method == "viterbi":
            self.decode = self.viterbi_decode
        else:
            self.decode = self.greedy_decode

        self.model = defaultdict(Counter)
        for word, freq in words.items():
            if len(word) == 1:
                self.model["s"][word] += freq
            else:
                self.model["b"][word[0]] += freq
                self.model["e"][word[-1]] += freq
                for c in word[1:-1]:
                    self.model["m"][c] += freq
        self.tags = "sbme"
        self.tags2id = {i:j for j,i in enumerate(self.tags)}
        self.log_total = {i:math.log(sum(self.model[i].values())) for i in tags}

    def find_word(self, sentence):
        # 根据最优路径进行分词
        if len(sentence) == 0:
            return
        scores = self.predict(sentence)
        path = self.decode(scores)
        buf = ""
        for v, c in zip(path, sentence):
            if v in [0, 1]:
                if buf:
                    yield buf
                buf = c
            else:
                buf += c
        if buf:
            yield buf

    def predict(self, sentence):
        # 逐字预测SBME标签的scores
        scores = np.zeros((len(sentence), 4))
        for i, c in enumerate(sentence):
            for j, k in self.model.items():
                score = np.log(k[c] + 0.0001) - self.log_total[j]
                scores[i][self.tags2id[j]] = score
        return scores

    def viterbi_decode(self, scores):
        # SBME标签序列解码最优路径
        return _viterbi_decode(scores, self.trans)

    def greedy_decode(self, scores):
        return np.argmax(scores, axis=1).tolist()

if __name__ == "__main__":
    import dataset
    import evaluation
    words, total = dataset.load_freq_words()
    tokenizer = HMMTokenizer(words, _log_trans)
    for text in dataset.load_sentences():
        print(hmm_tokenize(text))
        print(tokenizer.cut(text))

    text = dataset.load_human_history()
    evaluation.check_completeness(tokenizer.cut, text)
