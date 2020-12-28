from base import TokenizerBase

def backward_segment(text, words):
    # 逆向最长匹配，考虑词信息后置情况
    segments = []
    i = len(text) - 1
    while i >= 0:
        longest_word = text[i]
        for j in range(0, i):
            word = text[j:i+1]
            if word in words:
                if len(word) > len(longest_word):
                    longest_word = word
                    break
        segments.append(longest_word)
        i -= len(longest_word)
    # 因为是从后往前匹配，输出去需要逆转
    segments.reverse()
    return segments

class Tokenizer(TokenizerBase):

    def __init__(self, words):
        self.words = words

    def find_word(self, text):
        # 逆向最长匹配，考虑词信息后置情况
        segments = []
        i = len(text) - 1
        while i >= 0:
            longest_word = text[i]
            for j in range(0, i):
                word = text[j:i+1]
                if word in self.words:
                    if len(word) > len(longest_word):
                        longest_word = word
                        break
            segments.append(longest_word)
            i -= len(longest_word)
        # 因为是从后往前匹配，输出去需要逆转
        segments.reverse()
        yield from segments


if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    tokenizer = Tokenizer(words)
    for text in dataset.load_sentences():
        print(backward_segment(text, words))
        print(tokenizer.cut(text))

