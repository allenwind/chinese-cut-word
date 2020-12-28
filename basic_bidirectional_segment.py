from basic_forward_segment import forward_segment
from basic_backward_segment import backward_segment

def compute_single_chars(segments):
    # 计算单字符的次数
    return sum(1 for word in segments if len(word) == 1)

def bidirectional_segment(text, words):
    # 双向最长匹配
    segments1 = forward_segment(text, words)
    segments2 = backward_segment(text, words)
    if len(segments1) < len(segments2):
        return segments1
    elif len(segments1) > len(segments2):
        return segments2
    if compute_single_chars(segments1) < compute_single_chars(segments2):
        return segments1
    else:
        return segments2

if __name__ == "__main__":
    import dataset
    words, total = dataset.load_freq_words()
    for text in dataset.load_sentences():
        print(bidirectional_segment(text, words))

