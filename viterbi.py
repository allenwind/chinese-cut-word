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

class ViterbiDecoder:

    def __init__(self, trans):
        self.trans = trans

    def decode(self, scores):
        return viterbi_score(scores, self.trans)

if __name__ == "__main__":
    # testing
    scores = np.random.uniform(0, 1, size=(20, 4))
    trans = [[0.3, 0.7, 0.0, 0.0], 
             [0.0, 0.0, 0.3, 0.7], 
             [0.0, 0.0, 0.3, 0.7], 
             [0.3, 0.7, 0.0, 0.0]]
    path = viterbi_decode(scores, trans)
    print(path)
