import torch.nn as nn
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils.FaissIndexer import FaissIndex
import numpy as np

# Definir un modelo para convertir datos de EEG a embeddings
class EEGToEmbeddingModel(nn.Module):
    def __init__(self, eeg_dim, embedding_dim):
        super(EEGToEmbeddingModel, self).__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, eeg_dim),
            #nn.Conv1d(32, 64, 3),
            nn.ReLU(),
            nn.Linear(eeg_dim, embedding_dim)
        )
    
    def forward(self, eeg_input):
        return self.eeg_encoder(eeg_input)

def train_model_to_embedding(dataloader, eeg_data, word_embeddings, epochs, lr):
    eeg_dim = eeg_data.shape[1]
    embedding_dim = word_embeddings.shape[1]
    eeg_model = EEGToEmbeddingModel(eeg_dim, embedding_dim)

    # Definir la función de pérdida y el optimizador
    criterion = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=lr)

    # Entrenar el modelo de EEG a embedding
    eeg_model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            eeg_input, target_embeddings = batch
            optimizer.zero_grad()
            eeg_embeddings = eeg_model(eeg_input)
            target = torch.ones(len(eeg_embeddings))
            loss = criterion(eeg_embeddings, target_embeddings, target)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, MSE Loss: {loss.item()}')

    return eeg_model

def eval_model_as_classifier(model, x_test, y_test, unique_labels_embeddings, unique_labels):
    # Calcular el embedding de la entrada
    # Buscar el embeddings con menor similitud semántica con los embeddings de los labels
    # Se supone que si el label es el mismo embedding para todos.
    # Comparar el label real con el obtenido por el embedding
    # Si es el mismo, es un acierto, si no, no es un acierto.
    index = FaissIndex(np.array(unique_labels_embeddings), 768, unique_labels)
    count = 0
    succes_count = 0
    print("--")
    model.eval()
    for i in range(len(x_test)):
        count += 1
        y_pred = model(x_test[i])
        #label_pred = get_min_dist(y_pred, unique_labels_embeddings, unique_labels)
        label_pred = index.search(y_pred, 4)
        #print(f'Predicted labels: {label_pred} - Y_label: {y_test[i]}')
        #if label_pred[0] == y_test[i] or label_pred[1] == y_test[i] or label_pred[2] == y_test[i] or label_pred[3] == y_test[i]:
        if label_pred[0] == y_test[i]:
            succes_count += 1
    print(f'Count: {count}')
    print(f'Success count: {succes_count}')
    print(f'Accuracy: {succes_count/count}')


def get_min_dist(y_pred, unique_labels_embeddings, unique_labels):
    max_sim = 0
    index_max = -1 
    for i in range(len(unique_labels_embeddings)):
        sim = cosine_similarity([y_pred.detach().numpy()], [unique_labels_embeddings[i].detach().numpy()])
        if (sim > max_sim):
            max_sim = sim
            index_max = i
    return unique_labels[index_max]
    
