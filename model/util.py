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
        """
        Returns a list of words that are most similar to the passed-in embedding.
        """
        ans = sorted(embeddings_dict.keys(),
                      key=lambda word: spatial.distance.cosine(embeddings_dict[word], embedding))

        return ans[:k]


    def top_ten_hundred(self, embeddings_dict, answer, guess):
        """
        Adds evaluation for a single word onto th_tracker.
        guess is a word.
        """
        self.th_tracker['count'] += 1
        top_ten = self.find_closest_embeddings(embeddings_dict, embeddings_dict[answer], k = 10)
        if guess in top_ten:
            self.th_tracker['10'] += 1
            self.th_tracker['100'] += 1
        else:
            top_hundred = self.find_closest_embeddings(embeddings_dict, embeddings_dict[answer], k = 100)
            if guess in top_hundred:
                self.th_tracker['100'] += 1

    def compute_th_accuracy(self):
        accuracy = self.th_tracker['10'] / self.th_tracker['100']

        return accuracy


# e = Evaluator()
# embeddings_dict = e.load_glove_embeddings()
# print(e.find_closest_embeddings(embeddings_dict, embeddings_dict["will"]))


