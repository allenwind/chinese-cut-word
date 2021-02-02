import numpy as np
from base import TokenizerBase
from gensim.models import Word2Vec

# 根据相关矩阵进行无监督分词
# 待优化
# BERT

path = "/home/zhiwen/workspace/dataset/word2vec_baike/word2vec_baike"
model = Word2Vec.load(path)
# model.wv.vectors.shape
UNK = np.random.uniform(size=model.wv.syn0.shape[1])

def word2vec(word):
    if word not in model.wv.index2word:
        return UNK
    return model.wv.get_vector(word)

def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))

def tokenize(text, threshold=4.0):
    if not text:
        return []
    vectors = [UNK] + [word2vec(c) for c in text] + [UNK]
    words = [""]
    # 每个字只与相邻的词计算距离
    for i, c in enumerate(text, start=1):
        d1 = distance(vectors[i-1], vectors[i])
        d2 = distance(vectors[i], vectors[i+1])
        d = (d1 + d2) / 2
        print(d, c)
        if d <= threshold:
            words[-1] += c
        else:
            words.append(c)
    return words

class Tokenizer(TokenizerBase):
    
    def find_word(self, sentence):
        yield from tokenize(sentence)

if __name__ == "__main__":
    import dataset
    tokenizer = Tokenizer()
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))
