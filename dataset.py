import re
import glob
import collections
import itertools
import numpy as np

class WordTable(dict):

    def __missing__(self, key):
        return 1

    def word_logproba(self, word):
        pass

_HANS = "dataset/chinese-现代汉语词表.txt"
def load_chinese_words(file=_HANS, proba=False):
    with open(file, "r") as fp:
        lines = fp.read().splitlines()
    words = collections.defaultdict(int)
    for line in lines:
        word, _, freq = line.split("\t")
        words[word] += int(freq)
    total = sum(words.values())
    if proba:
        words = {i:j/total for i,j in words.items()}
    return words, total

_VOCAB = "dataset/Tencent_vocab.txt"
def load_tencent_words(file=_VOCAB):
    # 加载腾讯词表
    # extract from https://ai.tencent.com/ailab/nlp/en/embedding.html
    with open(file, "r") as fp:
        content = fp.read()
    words = set(content.splitlines())
    return words

_DICT = "dataset/dict.txt"
def load_freq_words(file=_DICT, proba=False, prefix=True):
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
        # 前缀字典
        if prefix:
            for i in range(len(word)):
                sw = word[:i+1]
                if sw not in words:
                    words[sw] = 0
    if proba:
        words = {i:j/total for i,j in words.items()}
    return words, total

_THUOCL = "/home/zhiwen/workspace/dataset/THUOCL中文分类词库/*.txt"
def load_THUOCL_words(path=_THUOCL, proba=False):
    # THUOCL中文分类词库
    files = glob.glob(path)
    words = collections.defaultdict(int)
    for file in files:
        with open(file, encoding="utf-8") as fd:
            lines = fd.readlines()
        for line in lines:
            try:
                word, freq = line.strip().split("\t")
                freq = freq.strip("?")
                words[word] += int(freq)
            except ValueError:
                print(line, file)
    total = sum(words.values())
    if proba:
        words = {i:j/total for i, j in words.items()}
    words = {i:j for i, j in words.items() if j > 0}
    return words, total

def load_all_words():
    # 加载所有词
    words = load_tencent_words()
    fwords, _ = load_freq_words()
    words.update(set(fwords))
    fwords, _ = load_THUOCL_words()
    words.update(set(fwords))
    return words

def load_sentences():
    # 测试分词效果的句子
    texts = []
    texts.append("守得云开见月明")
    texts.append("乒乓球拍卖完了")
    texts.append("无线电法国别研究")
    texts.append("广东省长假成绩单")
    texts.append("欢迎新老师生前来就餐")
    texts.append("上海浦东开发与建设同步")
    texts.append("独立自主和平等互利的原则")
    texts.append("黑天鹅和灰犀牛是两个突发性事件")
    texts.append("黄马与黑马是马，黄马与黑马不是白马，因此白马不是马。")
    texts.append("The quick brown fox jumps over the lazy dog.")
    texts.append("人的复杂的生理系统的特性注定了一件事情，就是从懂得某个道理到执行之间，是一个漫长的回路。")
    texts.append("除了导致大批旅客包括许多准备前往台北采访空难的新闻记者滞留在香港机场，直到下午2:17分日本亚细亚航空公司开出第一班离港到台北的班机才疏导了滞留在机场的旅客。")
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

_CTB6 = "dataset/ctb6_cws/"
def load_ctb6_cws(path=_CTB6, file="train.txt"):
    if not file.endswith(".txt"):
        file += ".txt"
    file = path + file
    # 复用load_icwb2_pku的加载方法
    return load_icwb2_pku(file)

_PEOPLE = "/home/zhiwen/desktop/people2014_cws/**/*.txt"
def load_people2014_cws(file=_PEOPLE):
    files = glob.glob()
    # TODO

def to_onehot(y, num_classes=4):
    categorical = np.zeros((len(y), num_classes))
    categorical[np.arange(len(y)), y] = 1
    return categorical

def build_sbme_tags(sentences, onehot=True):
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
                tags.append(0)
            else:
                tags.extend([1] + [2]*(len(word)-2) + [3])
        tags = np.array(tags)
        if onehot:
            tags = to_onehot(tags)
        y.append(tags)
        assert len("".join(sentence)) == len(tags)
    return y

class CharTokenizer:
    """简单字ID统计Tokenizer"""

    def __init__(self, mintf=5):
        self.char2id = {}
        self.MASK = 0
        self.UNK = 1
        self.mintf = mintf
        self.filters = set()

    def fit(self, X):
        # 建立词ID映射表
        chars = collections.defaultdict(int)
        for c in itertools.chain(*X):
            chars[c] += 1

        # 过滤低频词
        chars = {i: j for i, j in chars.items() \
                 if j >= self.mintf and i not in self.filters}
        # 0:MASK
        # 1:UNK
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i

    def transform(self, X):
        # 转成ID序列
        ids = []
        for sentence in X:
            s = []
            for char in sentence:
                s.append(self.char2id.get(char, self.UNK))
            ids.append(s)
        return ids

    @property
    def vocab_size(self):
        return len(self.char2id) + 2
