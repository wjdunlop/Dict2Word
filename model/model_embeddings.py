#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 4
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Vera Lin <veralin@stanford.edu>
"""

import torch.nn as nn
import numpy as np

def create_emb_layer(weights_matrix, src_pad_token_idx, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, src_pad_token_idx)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
    """

    def __init__(self, embed_size, vocab, glove_dict):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()


        self.embed_size = embed_size

        matrix_len = len(vocab)
        weights_matrix = np.zeros((matrix_len, 50))
        words_found = 0

        for i, word in enumerate(vocab):
            try:
                weights_matrix[i] = glove_dict[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(self.embed_size,))

        # default values
        self.source = None

        src_pad_token_idx = vocab.src['<pad>']

        self.source = create_emb_layer(weights_matrix, src_pad_token_idx, True)
        ### END YOUR CODE





