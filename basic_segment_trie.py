from base import TokenizerBase

_sentinel = object()

class Node:

    def __init__(self, value):
        self._children = {}
        self._value = value

    def _add_child(self, char, value, overwrite=False):
        child = self._children.get(char)
        if child is None:
            child = Node(value)
            self._children[char] = child
        elif overwrite:
            child._value = value
        return child

    def find(self, key):
        state = self
        for char in key:
            state = state._children.get(char)
            if state is None:
                break
        return state

class Trie(Node):
    """简单实现的Trie"""

    def __init__(self):
        super(Trie, self).__init__(_sentinel)

    def __contains__(self, key):
        return self[key] is not None

    def __getitem__(self, key):
        state = self
        for char in key:
            state = state._children.get(char)
            if state is None:
                return None
        return state._value

    def __setitem__(self, key, value):
        state = self
        for i, char in enumerate(key, start=0):
            if i < len(key) - 1:
                state = state._add_child(char, None, False)
            else:
                state = state._add_child(char, value, True)

    def __delitem__(self, key):
        self[key] = None

    def add(self, key):
        self[key] = _sentinel

    def pop(self, key):
        del self[key]

    def update(self, keys):
        for key in keys:
            self[key] = _sentinel

    def find_prefix(self, key):
        pass

def test_trie():
    import random
    import dataset
    trie = Trie()
    words = dataset.load_words()
    words = random.sample(words, k=100)
    trie.update(words)
    trie.update(["广东省", "长假", "成绩单"])
    assert "广东省" in trie

    trie.pop("成绩单")
    assert "成绩单" not in trie

    assert trie["长假"] is _sentinel

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

    def backward_segment(self, text):
        pass

if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    tokenizer = TrieTokenizer(words, fully=False)
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))

