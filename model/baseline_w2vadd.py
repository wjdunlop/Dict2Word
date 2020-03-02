from extract_data import convert_file_to_list
from util import Evaluator
import collections
import numpy as np

def create_definition_embedding(embeddings_dict, definitions_filename):
    """
    Creates and returns a list of definition embeddings by adding together the word vectors
    """
    definitions = convert_file_to_list(definitions_filename)

    def_embeddings = []

    for definition in definitions:
        definition = definition
        words = definition.split(' ')
        def_embedding = np.zeros_like(embeddings_dict['a'])

        for word in words:
            if word in embeddings_dict:
                def_embedding += embeddings_dict[word]

        def_embeddings.append(def_embedding)

    return def_embeddings


dummy_dict = {
              'a': np.array([1.,1.,1.]),
              'Stanford': np.array([1.,1.,1.]),
              'student': np.array([7.,1.,0.1]),
              'dorm': np.array([0.1,1.,100.]),
              'William': np.array([1.,0.1,0.1]),
              'Okada': np.array([0.1,0.,1.])
             }

for i in range(100):
    dummy_dict[str(i)] = np.array([0.1,float(i),0.1])

def_embeddings = create_definition_embedding(dummy_dict, '../data/dummy_definitions.txt')

words = convert_file_to_list('../data/dummy_words.txt')
guess_words = []

e = Evaluator()

for word in words:
    guess_word = e.find_closest_embeddings(dummy_dict, dummy_dict[word], k = 1)[0]

    e.top_ten_hundred(dummy_dict, word, guess_word)

print(e.th_tracker)
print(e.compute_th_accuracy())




