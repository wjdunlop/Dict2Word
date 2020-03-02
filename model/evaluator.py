import numpy as np
from scipy import spatial

class Evaluator:

    def __init__(self):
        # accuracy@10/100 = (th_tracker['10']/th_tracker['count'] / th_tracker['100']/th_tracker['count'])
        self.th_tracker = {'10': 0,
                           '100': 0,
                           'count': 0}


    def load_glove_embeddings(self, file = "../data/glove.6B.50d.txt", max_line = 10000):
        #currently we are only loading 10000 words because cosine similarity is too slow

        embeddings_dict = {}
        line_count = 0
        with open(file, 'r') as f:
            for line in f:
                if line_count >= max_line:
                    break
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
                line_count += 1
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
        guess is an embedding.
        """
        self.th_tracker['count'] += 1
        top_hundred = self.find_closest_embeddings(embeddings_dict, guess, k = 100)
        top_ten = top_hundred[:10]
        if answer in top_ten:
            self.th_tracker['10'] += 1
            self.th_tracker['100'] += 1
        elif answer in top_hundred:
            self.th_tracker['100'] += 1
        print(answer, top_ten)

    def compute_th_accuracy(self):
        return self.th_tracker['10'], self.th_tracker['100'], self.th_tracker['count']


# e = Evaluator()
# embeddings_dict = e.load_glove_embeddings()
# print(e.find_closest_embeddings(embeddings_dict, embeddings_dict["will"]))


