import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import progressbar
# using transformers
# used this tutorial
# https://medium.com/swlh/painless-fine-tuning-of-bert-in-pytorch-b91c14912caa
def getMaxLen(lines):
	# print('a')
	# print(lines.readlines()[:3])
	lines = list(lines)
	max1 = -1
	maxInd = -1
	c = 0
	for idx in range(len(lines)):
		line = lines[idx]
		size = len(line.split(' '))
		
		if size > max1:
			maxInd = idx
		# print(max1)
		max1 = max(max1, size)

		c += 1
		# print(line)
	# print(max1)
	# print(maxInd)
	# print(lines[maxInd])
	return max1

def defs_to_bert(dataset):

	print('Embedding dataset... pretrained BERT model [bert-base-uncased]')

	#Creating instance of BertModel
	bert_model = BertModel.from_pretrained('bert-base-uncased')

	#Creating intance of tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	#Specifying the max length
	
	print('gathering cls_head from file')
	
	print("Progress: ")
	
	outs = []
	path = 'data/data_train_definitions.txt'


	
	
	path = 'data/data_train_definitions.txt'
	file2 = open(path, encoding = 'utf-8')
	T = getMaxLen(file2)
	file = open(path, encoding = 'utf-8')
	# bar = progressbar.ProgressBar(max_value=len(file.readlines()))
	# print(file.readlines())
	i = 0
	for sentence in file.readlines()[:100]:
		print(i, ' ', len(file.readlines()))
		
		sentence = sentence[:-1]
		# print("poop")
		# print(sentence)
		#Step 1: Tokenize
		tokens = tokenizer.tokenize(sentence)

		#Step 2: Add [CLS] and [SEP]
		tokens = ['[CLS]'] + tokens + ['[SEP]']

		#Step 3: Pad tokens
		padded_tokens = tokens + ['[PAD]' for _ in range(T - len(tokens))]
		attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]

		#Step 4: Segment ids
		seg_ids = [0 for _ in range(len(padded_tokens))] #Optional!

		#Step 5: Get BERT vocabulary index for each token
		token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

		#Converting everything to torch tensors before feeding them to bert_model
		token_ids = torch.tensor(token_ids).unsqueeze(0) #Shape : [1, 12]
		attn_mask = torch.tensor(attn_mask).unsqueeze(0) #Shape : [1, 12]
		seg_ids   = torch.tensor(seg_ids).unsqueeze(0) #Shape : [1, 12]
			
		#Feed them to bert
		hidden_reps, cls_head = bert_model(token_ids, attention_mask = attn_mask,\
		                                  token_type_ids = seg_ids)
		# print(hidden_reps)
		#Out: torch.Size([1, 12, 768])
		# outs.append(cls_head)
		i += 1
		# progressbar.update(i)
	# print(outs)
	#Out: torch.Size([1, 768])


defs_to_bert(None)