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

def trans_humanize(trans):
    # 把转移矩阵转换为人类可读形式
    tags = "SBME"
    htrans = {}
    for i in range(4):
        for j in range(4):
            if trans[i][j] != 0.0:
                h = tags[i] + tags[j]
                htrans[h] = trans[i][j]
    return htrans

def trans2matrix(trans):
    # 转移矩阵的可读形式转化为矩阵形式
    matrix = np.zeros((4, 4))
    for i, s1 in enumerate("SBME"):
        for j, s2 in enumerate("SBME"):
            s = (s1 + s2)
            if s in trans:
                matrix[i][j] = trans[s]
    return matrix

def get_trans(T=1, log=True):
    # 转移矩阵一
    _trans1 = [[0.3, 0.7, 0.0, 0.0], 
               [0.0, 0.0, 0.3, 0.7], 
               [0.0, 0.0, 0.3, 0.7], 
               [0.3, 0.7, 0.0, 0.0]]

    # 转移矩阵二
    _trans2 = [[0.514, 0.486, 0.0, 0.0],
               [0.0, 0.0, 0.400, 0.600],
               [0.0, 0.0, 0.284, 0.716],
               [0.446, 0.554, 0.0, 0.0]]

    # 转移矩阵三
    _trans3 = [[0.300, 0.700, 0.0, 0.0],
               [0.0, 0.0, 0.400, 0.600],
               [0.0, 0.0, 0.284, 0.716],
               [0.446, 0.554, 0.0, 0.0]]

    # 转移矩阵四
    _trans4 = (np.array(_trans1) + np.array(_trans2)) / 2

    _trans5 = [[0.514, 0.486, 0.0, 0.0],
               [0.0, 0.0, 0.138, 0.862],
               [0.0, 0.0, 0.298, 0.702],
               [0.446, 0.554, 0.0, 0.0]]

    name = "_trans" + str(T)
    _trans = locals()[name]

    _trans = np.array(_trans)
    _trans = np.where(_trans==0, 0.0001, _trans)
    if log:
        _trans = np.log(_trans)
    return _trans

_log_trans = get_trans(T=2)

if __name__ == "__main__":
    # 测试
    import string
    # (seq_len, num_classes)
    scores = np.random.uniform(0, 1, size=(20, 4))
    tags = viterbi_decode(scores, _log_trans)
    print(tags)
    sentence = string.ascii_letters[:20]
    print(list(segment_by_tags(tags, sentence)))

    vd = ViterbiDecoder(_log_trans)
    print(list(vd.tokenize(sentence, scores)))

    print(trans_humanize(_log_trans))
