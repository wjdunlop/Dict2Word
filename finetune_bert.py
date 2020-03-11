import torch
import torch.nn as nn
from transformers import BertModel
import io

class ReverseDictionary(nn.Module):

    def __init__(self, embed_dim, hidden_dim, freeze_bert = False):
        super(SentimentClassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.lstm_fasttext = nn.LSTM(embed_dim, hidden_dim)
        self.lin_layer = nn.Linear(768 +hidden_dim, embed_dim)


    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        cont_reps, _ = self.bert_layer(seq, attention_mask = attn_masks)

        output, (cn, hn) = self.lstm_fasttext(seq)

        cls_rep = cont_reps[:, 0]

        toLinear = torch.concatenate([cls_rep, cn])

        #Obtaining the representation of [CLS] head
        
        #feed cls_rep to -> fasttext layer
        w_hat = self.lin_layer(toLinear)

        return w_hat

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


import numpy as np
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from nltk import word_tokenize

#from model_embeddings import ModelEmbeddings
from evaluator import Evaluator
from vocab import Vocab, VocabEntry
from utils import read_corpus, pad_sents, batch_iter


definition_indices = vocab.words2indices(definitions)
words_in = 0
words_out = 0

import timeit
start = timeit.default_timer()
losses = []

batch_size = 128

for epoch in range(500000):
    for src_sents, tgt_word in batch_iter(training_data, batch_size, False):
        model.zero_grad()
        x_lengths = [len(sent) for sent in src_sents]
        x = vocab.to_input_tensor(src_sents, "cpu")
        init_hidden = model.initHidden(len(src_sents), "cpu")
        tag_scores = model.forward(x, init_hidden, x_lengths)
        y_array = model.embedding.source[0](torch.tensor(vocab.words2indices(tgt_word))).double()
        y_pred = tag_scores[0].squeeze(dim = 1).double()
        loss = loss_function(y_pred, y_array, torch.tensor(1))
        loss.backward()
        optimizer.step() 
    losses.append(loss)
    print(epoch, loss, timeit.default_timer() - start)
    
stop = timeit.default_timer()

print('Time: ', stop - start)

import matplotlib.pyplot as plt
print(plt.plot([l.double() for l in losses]))