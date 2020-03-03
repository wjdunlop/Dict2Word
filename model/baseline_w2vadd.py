from extract_data import convert_file_to_list
from evaluator import Evaluator
import collections
import numpy as np
import nltk
from nltk.corpus import stopwords

def create_definition_embedding(embeddings_dict, definitions_filename):
    """
    Creates and returns a list of definition embeddings by adding together the word vectors
    """
    nltk.download('stopwords')
    stop_words = stopwords.words('english')

    negations = ["no", "not"]
    for word in negations:
        stop_words.remove(word)

    definitions = convert_file_to_list(definitions_filename)

    def_embeddings = []

    punctuation = ["\"","'",",",".",":",";"]

    for definition in definitions:
        definition = definition
        words = definition.split(' ')
        def_embedding = np.zeros_like(embeddings_dict['a'])

        for word in words:
            if word not in stop_words and word not in punctuation and word in embeddings_dict:
                def_embedding += embeddings_dict[word]


        def_embeddings.append(def_embedding)

    return def_embeddings

eval = Evaluator()
glove_dict = eval.load_glove_embeddings()


#TODO: we can use two dict of embeddings, one for definition (all words), one just for lexical items in the dictionary
validate_dict = {}
with open('../data/data_train_words.txt') as f:
    lines = f.readlines()
lines = [line[:-1] for line in lines]
words = sorted(list(set(lines)))
for word in words:
    validate_dict[word] = glove_dict[word]


def_embeddings = create_definition_embedding(glove_dict, "../data/data_train_definitions.txt")
print("loaded_embedding")



answers = []
with open('../data/data_train_words.txt') as f:
    answers += f.read().splitlines()
print("loaded_answers")

assert (len(def_embeddings) == len(answers))

for i in range(1000):
    eval.top_ten_hundred(validate_dict, answers[i], def_embeddings[i])
print("evaluated")

at10, at100, total = eval.compute_th_accuracy()
print(at10, at100, total)





#
# dummy_dict = {
#               'a': np.array([1.,1.,1.]),
#               'Stanford': np.array([1.,1.,1.]),
#               'student': np.array([7.,1.,0.1]),
#               'dorm': np.array([0.1,1.,100.]),
#               'William': np.array([1.,0.1,0.1]),
#               'Okada': np.array([0.1,0.,1.])
#              }
#
# for i in range(100):
#     dummy_dict[str(i)] = np.array([0.1,float(i),0.1])
#
# def_embeddings = create_definition_embedding(dummy_dict, '../data/dummy_definitions.txt')
#
# words = convert_file_to_list('../data/dummy_words.txt')
# guess_words = []
#
# e = Evaluator()
#
# for word in words:
#     guess_word = e.find_closest_embeddings(dummy_dict, dummy_dict[word], k = 1)[0]
#
#     e.top_ten_hundred(dummy_dict, word, guess_word)
#
# print(e.th_tracker)
# print(e.compute_th_accuracy())




