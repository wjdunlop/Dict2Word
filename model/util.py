import numpy as np
from scipy import spatial

class Evaluator:

    def __init__(self):
        # accuracy@10/100 = (th_tracker['10']/th_tracker['count'] / th_tracker['100']/th_tracker['count'])
        self.th_tracker = {'10': 0,
                           '100': 0,
                           'count': 0}


    def load_glove_embeddings(self, file = "../data/glove.6B.50d.txt"):
        embeddings_dict = {}
        with open(file, 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
        return embeddings_dict

    def find_closest_embeddings(self, embeddings_dict, embedding, k = 10):
        return sorted(embeddings_dict.keys(),
                      key=lambda word: spatial.distance.cosine(embeddings_dict[word], embedding)[:k])

    def top_ten_hundred(self, embeddings_dict, answer_embedding, guess_embedding):
        """
        Adds evaluation for a single word onto th_tracker.
        """
        self.th_tracker['count'] += 1
        top_ten = self.find_closest_embeddings(embeddings_dict, answer_embedding, k = 10)
        if guess_embedding in top_ten:
            self.th_tracker['10'] += 1
            self.th_tracker['100'] += 1
        else:
            top_hundred = self.find_closest_embeddings(embeddings_dict, answer_embedding, k = 100)
            if guess_embedding in top_hundred:
                self.th_tracker['100'] += 1


e = Evaluator()
embeddings_dict = e.load_glove_embeddings()
print(e.find_closest_embeddings(embeddings_dict, embeddings_dict["will"]))


