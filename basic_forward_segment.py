import dataset

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

if __name__ == "__main__":
    words = dataset.load_words()
    for text in dataset.load_sentences():
        print(forward_segment(text, words))

