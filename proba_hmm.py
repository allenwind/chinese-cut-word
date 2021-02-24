import math
from collections import Counter
from collections import defaultdict
import numpy as np

from viterbi import viterbi_decode as _viterbi_decode
from viterbi import get_trans
from base import TokenizerBase

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
                self.model["S"][word] += freq
            else:
                self.model["B"][word[0]] += freq
                self.model["E"][word[-1]] += freq
                for c in word[1:-1]:
                    self.model["M"][c] += freq
        self.tags = "SBME"
        self.tags2id = {i:j for j,i in enumerate(self.tags)}
        self.log_total = {i:math.log(sum(self.model[i].values())) for i in self.tags}

    def find_word(self, sentence):
        # 根据最优路径进行分词
        if len(sentence) == 0:
            return
        scores = self.predict(sentence)
        tags = self.decode(scores)
        buf = ""
        for v, c in zip(tags, sentence):
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
    _log_trans = get_trans(T=2, log=True)

    tokenizer = HMMTokenizer(words, _log_trans)
    for text in dataset.load_sentences():
        # print(hmm_tokenize(text))
        print(tokenizer.cut(text))

    text = dataset.load_human_history()
    evaluation.check_completeness(tokenizer.cut, text)
