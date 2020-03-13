#!/usr/bin/env python
# coding: utf-8

# In[11]:


from transformers import BertModel, BertTokenizer
import io

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
import pickle
import timeit
from scipy import spatial
import time
from evaluator import Evaluator
from vocab import Vocab, VocabEntry
from utils import read_corpus, pad_sents, batch_iter_bert
from datetime import datetime

ts = time.time()
datetm = datetime.fromtimestamp(ts)
dt = str(datetm)[:-7]
print(dt)
dt = dt.replace(' ', '_')
dt = dt.replace(':', '-')
print(dt)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('use device: %s' % device)


# In[12]:


words, defs, ft_dict = pickle.load( open( "../data/words_defs_dict_1M.train", "rb" ))

vocab = VocabEntry.from_corpus(defs, 1000000, 0)
for w in ft_dict:
    vocab.add(w)


# In[13]:


def create_emb_layer(weights_matrix, src_pad_token_idx, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, src_pad_token_idx)
    emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix)) #figure out what is here
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """

    def __init__(self, embed_size, vocab, fasttext_dict):
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
        #print(len(vocab), weights_matrix.shape)
        for word, index in vocab.word2id.items():
            try:
                weights_matrix[index] = np.array(fasttext_dict[word])
                words_found += 1
            except KeyError:
                weights_matrix[index] = np.random.normal(scale=0.6, size=(self.embed_size,))

        # default values
        src_pad_token_idx = vocab['<pad>']
        self.source = create_emb_layer(weights_matrix, src_pad_token_idx, True)


# In[14]:


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


# In[15]:


class ReverseDictionary(nn.Module):

    def __init__(self, embed_dim, hidden_dim, vocab, ft_dict, freeze_bert = False):
        super(ReverseDictionary, self).__init__()
        #Instantiating BERT model object 
        
        self.ft_embedding = ModelEmbeddings(embed_dim, vocab, ft_dict)
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
#         Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.lstm_fasttext = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
        self.lin_layer = nn.Linear(hidden_dim+768, embed_dim)


    def forward(self, ft_input, lengths, bert_input, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        embedded = self.ft_embedding.source[0](ft_input).transpose(1,0)

        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        output, (cn, hn) = self.lstm_fasttext(embedded)
        
        cont_reps, _ = self.bert_layer(bert_input, attention_mask = attn_masks)
        cls_rep = cont_reps[:, 0]

        toLinear = torch.cat([cls_rep, cn.squeeze(0)], 1)

        #Obtaining the representation of [CLS] head
        
        #feed cls_rep to -> fasttext layer
        projected = self.lin_layer(toLinear)

        return projected


# In[17]:


model = ReverseDictionary(300, 300, vocab, ft_dict)
if device is 'cuda:0':
    model.cuda()
loss_function = nn.L1Loss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr = .001)


# In[18]:


int_sents = vocab.words2indices(defs)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_len = max(len(x) for x in int_sents)
sents_ft_ids = int_sents
sents_bert_ids = []
masks = []
for d in defs:
    tokens = ['[CLS]'] + d + ['[SEP]']
    padded_tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    seg_ids = [0 for _ in range(len(padded_tokens))]
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    sents_bert_ids.append(token_ids)
    masks.append(attn_mask)
    

assert(len(sents_bert_ids) == len(defs))
assert(len(sents_bert_ids) == len(masks))
assert(len(masks) == len(words))
data = [(defs[i], len(defs[i]), sents_bert_ids[i], masks[i], words[i]) for i in range(len(defs))]


# In[19]:


start = timeit.default_timer()
losses = []
BATCH_SIZE = 2
best_loss = 100000
for epoch in range(1):
    print('epoch: ', epoch)
    loss_cum = []
    for batch_defs, batch_lengths, batch_bert_ids, batch_masks, batch_words in  \
                       batch_iter_bert(data, BATCH_SIZE, shuffle=False):
        
        model.zero_grad()
        #print(batch_lengths)
        batch_ft_ids = vocab.to_input_tensor(batch_defs, device = device)
        batch_bert_ids = torch.tensor(batch_bert_ids, device = device)
        batch_masks = torch.tensor(batch_masks, device = device)
        tag_scores = model.forward(batch_ft_ids, batch_lengths, batch_bert_ids, batch_masks)
        
        y_pred = tag_scores[0].double()
        y_indices = torch.tensor([vocab[i] for i in batch_words])
        y_array = model.ft_embedding.source[0](y_indices).double()
        #print(y_array.shape)
        print("lll")
        print(y_pred.shape)
        print(y_array.shape)
        loss = loss_function(y_pred, y_array)
        print(loss)
        loss_cum.append(loss)
        loss.backward()
        optimizer.step() 

    
    eloss = sum(loss_cum)/len(loss_cum)
    if eloss < best_loss:
        print("new best loss")
        best_loss = eloss
        best_model = model
    torch.save(best_model.state_dict(), 'model'+dt+'.pt')
    lossavg = sum(loss_cum)/len(loss_cum)
    losses.append(loss)
    #print(epoch, lossavg, timeit.default_timer() - start)


# In[20]:


eval = Evaluator()
model.zero_grad()

for i in range(len(words)):
    # model.zero_grad()
    # tag_scores = model.forward(sents_bert_id[i], sents_ft_id[i], masks[i])
    # y_pred = tag_scores[0].double()#.unsqueeze(1)
    # #print(y_pred)
    # y_array = model.ft_embedding.source[0](torch.tensor(vocab[words[i]])).double().unsqueeze(1)
    # #print(y_array)
    # #print(y_pred.shape, y_array.shape)
    # loss = loss_function(y_pred, y_array)
    eval.top_ten_hundred(ft_dict, words[i], y_pred.detach().numpy())
    print(np.linalg.norm(ft_dict[words[i]]-y_pred.detach().numpy()))
#     print(np.linalg.norm(ft_dict[words[i]]-y_pred.detach().numpy()))
#     print(sorted(ft_dict.keys(), key=lambda word: spatial.distance.cosine(ft_dict[word], y_pred.detach().numpy())))
#     print(ft_dict['fault'].shape, y_pred.detach().numpy().shape)
#     print(loss)


# In[ ]:





# In[ ]:




