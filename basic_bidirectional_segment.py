import dataset
from basic_forward_segment import forward_segment
from basic_backward_segment import backward_segment

def count_single_char(segments):
    return sum(1 for word in segments if len(word) == 1)

def bidirectional_segment(text, words):
    # 双向最长匹配
    fsegments = forward_segment(text, words)
    bsegments = backward_segment(text, words)
    if len(fsegments) < len(bsegments):
        return fsegments
    elif len(fsegments) > len(bsegments):
        return bsegments
    if count_single_char(fsegments) < count_single_char(bsegments):
        return fsegments
    else:
        return bsegments

if __name__ == "__main__":
    words = dataset.load_words()
    for text in dataset.load_sentences():
        print(bidirectional_segment(text, words))

