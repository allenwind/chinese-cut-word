from base import TokenizerBase
from datastructure import Trie

class TrieTokenizer(TokenizerBase):
    """把词图构建在Trie树上，当词表很大时Python的实现比较慢且耗内存"""

    def __init__(self, words, fully=False):
        self.trie = Trie()
        self.trie.update(words)
        self.fully = fully

    def find_word(self, sentence):
        if self.fully:
            words = self.fully_segment(sentence)
        else:
            words = self.forward_segment(sentence)
        for word in words:
            yield word[0]

    def fully_segment(self, text):
        words = []
        size = len(text)
        for i in range(size):
            state = self.trie
            for j in range(i, size):
                state = state.find(text[j])
                if state:
                    if state._value is not None:
                        words.append((text[i: j+1], state._value, i, j+1))
                else:
                    break
        return words

    def forward_segment(self, text):
        # 原理可参看basic_forward_segment.py
        i = 0
        size = len(text)
        words = []
        while i < size:
            state = self.trie.find(text[i])
            if state:
                j = i + 1
                k = j
                value = state._value
                for j in range(i+1, size):
                    state = state.find(text[j])
                    if not state:
                        break
                    if state._value is not None:
                        value = state._value
                        k = j + 1
                if value is not None:
                    words.append((text[i:k], value, i, k))
                    i = k - 1
            i += 1
        return words

if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    tokenizer = TrieTokenizer(words, fully=False)
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))

