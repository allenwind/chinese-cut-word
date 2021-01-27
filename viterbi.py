import numpy as np

def viterbi_decode(scores, trans, return_score=False):
    # 使用viterbi算法求最优路径
    # scores.shape = (seq_len, num_tags)
    # trans.shape = (num_tags, num_tags)
    dp = np.zeros_like(scores)
    backpointers = np.zeros_like(scores, dtype=np.int32)
    dp[0] = scores[0]
    for t in range(1, scores.shape[0]):
        v = np.expand_dims(dp[t-1], axis=1) + trans
        dp[t] = scores[t] + np.max(v, axis=0)
        backpointers[t] = np.argmax(v, axis=0)

    viterbi = [np.argmax(dp[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    if return_score:
        viterbi_score = np.max(dp[-1])
        return viterbi, viterbi_score
    return viterbi

def segment_by_tags(tags, sentence):
    # 通过SBME序列对sentence分词
    assert len(tags) == len(sentence)
    buf = ""
    for t, c in zip(tags, sentence):
        # t is S or B
        if t in [0, 1]:
            if buf:
                yield buf
            buf = c
        # t is M or E
        else:
            buf += c
    if buf:
        yield buf

class ViterbiDecoder:

    def __init__(self, trans):
        self.trans = trans

    def decode(self, scores):
        return viterbi_decode(scores, self.trans)

    def tokenize(self, sentence, scores):
        tags = self.decode(scores)
        yield from segment_by_tags(tags, sentence)

if __name__ == "__main__":
    # testing
    import string
    # (seq_len, num_classes)
    scores = np.random.uniform(0, 1, size=(20, 4))
    trans = [[0.3, 0.7, 0.0, 0.0], 
             [0.0, 0.0, 0.3, 0.7], 
             [0.0, 0.0, 0.3, 0.7], 
             [0.3, 0.7, 0.0, 0.0]]
    tags = viterbi_decode(scores, trans)
    print(tags)
    sentence = string.ascii_letters[:20]
    print(list(segment_by_tags(tags, sentence)))

    vd = ViterbiDecoder(trans)
    print(list(vd.tokenize(sentence, scores)))
