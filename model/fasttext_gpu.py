#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
from collections import namedtuple
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torchnlp.nn import Attention
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from nltk import word_tokenize
import pickle
from model_embeddings import ModelEmbeddings
from evaluator import Evaluator
from vocab import Vocab, VocabEntry
from utils import read_corpus, pad_sents, batch_iter
import pickle
import timeit
import time
from datetime import datetime


def create_emb_layer(weights_matrix, src_pad_token_idx, device = "cpu", non_trainable=True):
		num_embeddings, embedding_dim = weights_matrix.shape
		emb_layer = nn.Embedding(num_embeddings, embedding_dim, src_pad_token_idx)
		emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix).float().to(device)) #figure out what is here
		if non_trainable:
			emb_layer.weight.requires_grad = False
		return emb_layer, num_embeddings, embedding_dim

class ModelEmbeddings(nn.Module): 
	"""
	Class that converts input words to their embeddings.
	"""

	def __init__(self, embed_size, vocab, fasttext_dict, device):
		"""
		Init the Embedding layers.

		@param embed_size (int): Embedding size (dimensionality)
		@param vocab (VocabEntry)
		"""
		super(ModelEmbeddings, self).__init__()

		self.embed_size = embed_size

		matrix_len = len(vocab)
		weights_matrix = np.zeros((matrix_len, self.embed_size))
		words_found = 0

		# for word, index in vocab.word2id.items():
		#     weights_matrix[index] = np.array(fasttext_model.get_word_vector(word))
		
		for word, index in vocab.word2id.items():
			try:
				weights_matrix[index] = np.array(fasttext_dict[word])
				words_found += 1
			except KeyError:
				weights_matrix[index] = np.random.normal(scale=0.6, size=(self.embed_size,))

		# default values
		src_pad_token_idx = vocab['<pad>']
		self.source = create_emb_layer(weights_matrix, src_pad_token_idx, device, True)

class LSTMModel(nn.Module):
	def __init__(self, input_size, hidden_size, vocab, fasttext_model, device = 'cpu'):
		super(LSTMModel, self).__init__()
		self.hidden_size = hidden_size
		self.input_size = input_size
		self.vocab = vocab
		self.embedding = ModelEmbeddings(input_size, vocab, fasttext_model, device)
		self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = True)
		self.linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias = True)
		self.linear2 = nn.Linear(self.hidden_size, self.hidden_size, bias = True)
		self.attention = Attention(self.hidden_size)

	def forward(self, input_, hidden, lengths, dropout_rate = 0.3):
		embedded = self.embedding.source[0](input_)
		embedded = pack_padded_sequence(embedded, lengths)
		output, (h_n, c_n) = self.lstm(embedded)   
		hidden_permuted = h_n.contiguous().view(1, -1, self.hidden_size * 2).permute(1,0,2)
		 
		projected = self.linear(hidden_permuted)
		dropout = nn.Dropout(dropout_rate)
		projected_dropped = dropout(projected)
		projected2 = self.linear2(projected_dropped)
		return projected2, hidden

	def initHidden(self, batch_size, device = None):
		return torch.zeros(1, batch_size, self.hidden_size, device = device)

def main(argv):	
	EPOCHS = int(argv[0])
	BATCH_SIZE = int(argv[1])
	LEARNING_RATE = float(argv[2])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	words = []
	definitions = []
	sub_fasttext_dict = {}
	with open("../data/words_defs_dict.train", "br") as f:
		words, definitions, sub_fasttext_dict = pickle.load(f)
	print("number of words:", len(words))

	eval = Evaluator()
	vocab = VocabEntry.from_corpus(definitions, 1000000, 0)
	for w in words:
		vocab.add(w)
	print("vocab length:", len(vocab))

	assert(len(words) == len(definitions))
	training_data = [(definitions[i], words[i]) for i in range(len(words))]

	model = LSTMModel(100, 100, vocab, sub_fasttext_dict, device)
	model.to(device)
	loss_function = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
	optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

	dt = str(datetime.fromtimestamp(time.time()))[:-7]
	print(dt)
	dt = dt.replace(' ', '_')
	dt = dt.replace(':', '-')
	start = timeit.default_timer()
	losses = []

	best_loss = float('inf')
	for epoch in range(EPOCHS):
		epoch_losses = []
		count = 0
		for src_sents, tgt_word in batch_iter(training_data, BATCH_SIZE, False):
			model.zero_grad()
			x_lengths = [len(sent) for sent in src_sents]
			x = vocab.to_input_tensor(src_sents, device).to(device)
			init_hidden = model.initHidden(len(src_sents), device)
			tag_scores = model.forward(x, init_hidden, x_lengths)
			
			y_indices = vocab.words2indices(tgt_word)
			y_array = model.embedding.source[0](torch.tensor(y_indices, device = device)).double()
			y_pred = tag_scores[0].squeeze(dim = 1).double()
			y_match = torch.ones(y_pred.shape[0])

			loss = loss_function(y_pred, y_array, y_match)

			loss.backward()
			optimizer.step()
			count += 1
			epoch_losses.append(loss)
			if count % 200 == 0:
				print("Time elapsed", timeit.default_timer() - start, "Epoch", epoch, " Count", count, ": Loss", loss)
				losses.append(loss)
		eloss = sum(epoch_losses)/len(epoch_losses)
		if eloss < best_loss:
			best_loss = eloss
			title = 'ft_model'+ dt +'.pt'
			torch.save(model.state_dict(), title)
			print("model saves as:", title, "with epoch loss of ", eloss)
		
	stop = timeit.default_timer()

	print('Time: ', stop - start)

	import matplotlib.pyplot as plt
	print(plt.plot([l.double() for l in losses]))

if __name__ == "__main__":
   main(sys.argv[1:])
