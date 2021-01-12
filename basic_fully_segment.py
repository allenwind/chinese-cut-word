from base import TokenizerBase

def fully_segment(text, words):
    # 完全切分，常用在搜索引擎或细粒度场景上
    # 该方法不满足分词的完整性
    segments = []
    size = len(text)
    # 扫描(i<j)，如果在词典中就当做一个词
    for i in range(size): 
        for j in range(i+1, size+1):
            word = text[i:j]
            if word in words:
                segments.append(word)
    return segments

class Tokenizer(TokenizerBase):

    def __init__(self, words):
        self.words = words

    def find_word(self, sentence):
        # bug?
        yield from fully_segment(sentence, self.words)

if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    tokenizer = Tokenizer(words)
    for text in dataset.load_sentences():
        print(fully_segment(text, words))
        print(tokenizer.cut(text))
