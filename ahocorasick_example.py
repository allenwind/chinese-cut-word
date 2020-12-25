import math
import dataset
import ahocorasick

# ahocorasicks使用例子

words, _ = dataset.load_freq_words(proba=True)
logtotal = math.log(sum(words.values()))
am = ahocorasick.Automaton()
for word, proba in words.items():
    logproba = (math.log(proba) if proba > 0 else 0) - logtotal
    am.add_word(word, (word, proba))

am.make_automaton()

text = "黑天鹅和灰犀牛是两个突发性事件"
for i, (j, k) in am.iter(text):
    print(i, j, k)

