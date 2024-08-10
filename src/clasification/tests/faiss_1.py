import pandas as pd
import numpy as np
#
import faiss
import pickle

word_file = open("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/word_embeddings", 'rb')
words_embeddings = pickle.load(word_file)


d = 768
indexFlat = faiss.IndexFlatL2(d)
indexFlat.add(words_embeddings)
indexFlat.ntotal
indexFlat.is_trained


# Ejemplo de recuperación
k = 4
query_vector = np.array([words_embeddings[1]])
#%time
DFlat, rsFlat = indexFlat.search(query_vector, k)  # Búsqueda

print(rsFlat)

