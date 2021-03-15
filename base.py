import re

# TODO超长文本的并行分词扩展

class TokenizerBase:
    """分词的基类，继承该类并在find_word实现分词的核心算法"""

    spaces = re.compile("(\r\n|\s)", re.U)
    english = re.compile("[a-zA-Z0-9]", re.U)
    chinese = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)

    def cut(self, text):
        return list(self._cut(text))

    def _cut(self, text):
        # 把长文本切分为句子块
        for block in self.chinese.split(text):
            if not block:
                continue
            if self.chinese.match(block):
                yield from self.cut_block(block)
            else:
                for s in self.spaces.split(block):
                    if self.spaces.match(s):
                        yield s
                    else:
                        yield from s

    def cut_block(self, sentence):
        # 对文本进行分块分句后分词
        buf = ""
        for word in self.find_word(sentence):
            if len(word) == 1 and self.english.match(word):
                buf += word
            else:
                if buf:
                    yield buf
                    buf = ""
                yield word
        if buf:
            yield buf

    def find_word(self, sentence):
        # 从这里实现分词算法的核心
        # 从句子中发现可以构成的词，返回可迭代对象
        raise NotImplementedError

if __name__ == "__main__":
    # 简单的测试
    import dataset

    class Tokenizer(TokenizerBase):

        def find_word(self, sentence):
            for word in sentence:
                yield word

    tokenizer = Tokenizer()
    for text in dataset.load_sentences():
        print(tokenizer.cut(text))
