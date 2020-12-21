import dataset

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

if __name__ == "__main__":
    words = dataset.load_words()
    for text in dataset.load_sentences():
        print(backward_segment(text, words))

