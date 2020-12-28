from base import TokenizerBase

def forward_segment(text, words):
    # 正向最长匹配
    # 在fully_segment基础上只匹配最长的词
    segments = []
    size = len(text)
    i = 0
    while i < size:
        # 从位置i的单字出发
        longest_word = text[i]
        for j in range(i+1, size+1):
            # 寻找比这个单字更长的词
            word = text[i:j]
            if word in words:
                if len(word) > len(longest_word):
                    longest_word = word
        segments.append(longest_word)
        i += len(longest_word)
    return segments

def forward_segment2(text, words):
    # forward_segment另外一种实现
    if not text:
        return []
    segments = [""]
    for c in text:
        # 临时词
        word = segments[-1] + c
        if word in words:
            segments[-1] += c
        else:
            segments.append(c)
    return segments

class Tokenizer(TokenizerBase):

    def __init__(self, words):
        self.words = words

    def find_word(self, text):
        # 正向最长匹配
        # 在fully_segment基础上只匹配最长的词
        size = len(text)
        i = 0
        while i < size:
            # 从位置i的单字出发
            longest_word = text[i]
            for j in range(i+1, size+1):
                # 寻找比这个单字更长的词
                word = text[i:j]
                if word in self.words:
                    if len(word) > len(longest_word):
                        longest_word = word
            yield longest_word
            i += len(longest_word)

if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    tokenizer = Tokenizer(words)
    for text in dataset.load_sentences():
        print(forward_segment(text, words))
        print(forward_segment2(text, words))
        print(tokenizer.cut(text))



