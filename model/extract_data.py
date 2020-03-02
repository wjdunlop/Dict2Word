"""
extracts word definitions from data files.
"""
from tqdm import tqdm
import time

words = []
definitions = []

def extract_definitions(filename):
    # Using readlines() 
    data_raw = open(filename, 'r') 
    for line in tqdm(data_raw): 
        line_items = line.split('\t')

        words.append(line_items[0])
        definitions.append(line_items[3])

    # Closing files 
    data_raw.close()

extract_definitions('../data/data_train_raw.txt')

with open('../data/data_train_definitions.txt', 'w') as filehandle_d:
    for definition in definitions:
        filehandle_d.write('%s' % definition)

with open('../data/data_train_words.txt', 'w') as filehandle_w:
    for word in words:
        filehandle_w.write('%s\n' % word)