import glob
import re
import math
import collections
from base import TokenizerBase
import ahocorasick # pip3 install pyahocorasick

class AutomatonTokenizer(TokenizerBase):
    """把词图构建在AC自动机上，并实现各种分词算法"""

    def __init__(self, words, algorithm="forward_segment"):
        self.am = ahocorasick.Automaton()
        logtotal = math.log(sum(words.values()))
        for word, proba in words.items():
            logproba = (math.log(proba) if proba > 0 else -1e6) - logtotal
            self.am.add_word(word, (word, logproba))
        self.am.make_automaton()
        self.algorithm = algorithm

    def find_word(self, sentence):
        if self.algorithm == "forward_segment":
            yield from self.forward_segment(sentence)
        elif self.algorithm == "backward_segment":
            yield from self.backward_segment(sentence)
        elif self.algorithm == "min_words_segment":
            yield from self.min_words_segment(sentence)
        elif self.algorithm == "max_proba_segment":
            yield from self.max_proba_segment(sentence)
        else:
            yield from self.fully_segment(sentence)

    def fully_segment(self, text):
        # 全模式，原理参考本repo中的basic_fully_segment.py
        words = []
        for i, (word, proba) in self.am.iter(text):
            words.append(word)
        return words

    def forward_segment(self, text):
        # 正向最长匹配，参考forward_segment.py中的两种实现
        segments = []
        size = len(text)
        i = 0
        while i < size:
            # 从位置i的单字出发
            longest_word = text[i]
            for j in range(i+1, size+1):
                # 寻找比这个单字更长的词
                word = text[i:j]
                if self.am.match(word):
                    if len(word) > len(longest_word):
                        longest_word = word
            segments.append(longest_word)
            i += len(longest_word)
        return segments

    def backward_segment(self, text):
        # 逆向最长匹配，考虑词信息后置情况
        segments = []
        i = len(text) - 1
        while i >= 0:
            longest_word = text[i]
            for j in range(0, i):
                word = text[j:i+1]
                if self.am.match(word):
                    if len(word) > len(longest_word):
                        longest_word = word
                        break
            segments.append(longest_word)
            i -= len(longest_word)
        # 因为是从后往前匹配，输出去需要逆转
        segments.reverse()
        return segments

if __name__ == "__main__":
    import dataset
    import evaluation
    words, total = dataset.load_freq_words(proba=True, prefix=False)
    tokenizer = AutomatonTokenizer(words, algorithm="forward_segment")
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))

    # 测试分词的完整性
    text = dataset.load_human_history()
    words = tokenizer.cut(text)
    assert "".join(words) == text

    evaluation.evaluate_speed(tokenizer.cut, text, rounds=5)
