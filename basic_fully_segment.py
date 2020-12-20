import dataset

def fully_segment(text, words):
    # 完全切分，常用在搜索引擎或细粒度场景上
    segments = []
    size = len(text)
    # 扫描(i<j)，如果在词典中就当做一个词
    for i in range(size): 
        for j in range(i+1, size+1):
            word = text[i:j]
            if word in words:
                segments.append(word)
    return segments

if __name__ == "__main__":
    words = dataset.load_words()
    for text in dataset.load_sentences():
        print(fully_segment(text, words))

