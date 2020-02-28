import numpy as np
from scipy import spatial


def load_glove_embeddings(file = "../data/glove.6B.50d.txt"):
    embeddings_dict = {}
    with open(file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def find_closest_embeddings(embeddings_dict, embedding, k = 10):
    return sorted(embeddings_dict.keys(),
                  key=lambda word: spatial.distance.cosine(embeddings_dict[word], embedding)[:k])

embeddings_dict = load_glove_embeddings()
print(find_closest_embeddings(embeddings_dict, embeddings_dict["will"]))


