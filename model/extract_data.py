"""
extracts word definitions from data files.
"""
from tqdm import tqdm
import time


def extract_definitions(filename):
    # Using readlines() 
    words = []
    definitions = []

    data_raw = open(filename, 'r')
    for line in tqdm(data_raw): 
        line_items = line.split('\t')

        words.append(line_items[0])
        definitions.append(line_items[3])

    # Closing files 
    data_raw.close()

# extract_definitions('../data/data_train_raw.txt')

# with open('../data/data_train_definitions.txt', 'w') as filehandle_d:
#     for definition in definitions:
#         filehandle_d.write('%s' % definition)

# with open('../data/data_train_words.txt', 'w') as filehandle_w:
#     for word in words:
#         filehandle_w.write('%s\n' % word)

def convert_file_to_list(filename):
    file = open(filename, 'r')
    ls = []

    for line in tqdm(file): 
        ls.append(line.strip())

    # Closing files 
    file.close()

    return ls