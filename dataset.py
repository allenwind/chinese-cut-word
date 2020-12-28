import re
import collections
import itertools

class WordTable(dict):

    def __missing__(self, key):
        return 1

    def word_logproba(self, word):
        pass

_VOCAB = "dataset/Tencent_vocab.txt"
def load_tencent_words(file=_VOCAB):
    # 加载腾讯词表
    # extract from https://ai.tencent.com/ailab/nlp/en/embedding.html
    with open(file, "r") as fp:
        content = fp.read()
    words = set(content.splitlines())
    return words

_DICT = "dataset/dict.txt"
def load_freq_words(file=_DICT, proba=False):
    # 词频表
    words = {}
    total = 0
    with open(file, "r") as fp:
        lines = fp.read().splitlines()
    for line in lines:
        word, freq = line.split(" ")[:2]
        freq = int(freq)
        words[word] = freq
        total += freq
        for i in range(len(word)):
            sw = word[:i+1]
            if sw not in words:
                words[sw] = 0
    if proba:
        words = {i:j/total for i,j in words.items()}
    return words, total

_THUOCL = "/home/zhiwen/workspace/dataset/THUOCL中文分类词库/*.txt"
def load_THUOCL_words(path=_THUOCL, proba=True):
    # THUOCL中文分类词库
    files = glob.glob(path)
    words = collections.defaultdict(int)
    for file in files:
        with open(file, encoding="utf-8") as fd:
            lines = fd.readlines()
        for line in lines:
            try:
                word, freq = line.strip().split("\t")
                words[word] += int(freq)
            except ValueError:
                print(line, file)
    if proba:
        total = sum(words.values())
        words = {i:j/total for i, j in words.items()}
    words = {i:j for i, j in words.items() if j > 0}
    return words

def load_all_words():
    pass

def load_sentences():
    # 测试分词效果的句子
    texts = []
    texts.append("守得云开见月明")
    texts.append("广东省长假成绩单")
    texts.append("The quick brown fox jumps over the lazy dog")
    texts.append("黑天鹅和灰犀牛是两个突发性事件")
    texts.append("欢迎新老师生前来就餐")
    texts.append("独立自主和平等互利的原则")
    texts.append("人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。")
    return texts

_HUMAH = "/home/zhiwen/workspace/dataset/human-history人类简史-从动物到上帝.txt"
def load_human_history(file=_HUMAH):
    # 加载长文本
    with open(file, "r") as fp:
        text = fp.read()
    return text

_PKU = "/home/zhiwen/workspace/dataset/icwb2-data/training/msr_training.utf8"
def load_icwb2_pku(file=_PKU):
    with open(file, "r") as fp:
        text = fp.read()
    sentences = text.splitlines()
    sentences = [re.split("\s+", sentence) for sentence in sentences]
    sentences = [[w for w in sentence if w] for sentence in sentences]
    return sentences

def build_sbme_tags(sentences):
    # 0: s单字词
    # 1: b多字词首字
    # 2: m多字词中间
    # 3: e多字词末字
    id2tag = {0:"s", 1:"b", 2:"m", 3:"e"}
    y = []
    for sentence in sentences:
        tags = []
        for word in sentence:
            if len(word) == 1:
                tags.append(o)
            else:
                y.extend([1] + [2]*(len(word)-2) + [3])
        y.append(tags)
    return y
