import faiss
import pickle
import torch
import numpy as np
# word_file = open("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/word_embeddings", 'rb')
# words_embeddings = pickle.load(word_file)
# d = 768

class FaissIndex:
    #def __init__(self, word_embeddings_path, d):
    def __init__(self, word_embeddings, d, words):
        self.index = faiss.IndexFlatL2(d)
        self.index.add(word_embeddings)
        self.words = words
        print(f'Init faiss index - Total: {self.index.ntotal}') 
        self.index.is_trained

    def search(self, query_vector, k):
        query = np.array([query_vector.detach().numpy()])
        DFlat, rsFlat = self.index.search(query, k)
        return [self.words[i] for i in rsFlat[0]]
