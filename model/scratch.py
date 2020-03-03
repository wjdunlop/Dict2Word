from nltk.corpus import stopwords
import nltk
train_dict = {}
with open('../data/data_train_words.txt') as f:
	lines = f.readlines()
lines = [line[:-1] for line in lines]
lines = sorted(list(set(lines)))

print(lines)