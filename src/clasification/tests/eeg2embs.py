import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Model
import pickle
import os

# Custom dataset
class EEGTextDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = eeg_data
        self.labels = labels
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]

def filter_missing_data(data_frame):
    return data_frame.dropna()

# Returns features and labels
def load_data(filename):
    data_frame = pd.read_csv(filename)
    data_frame = filter_missing_data(data_frame)
    word_labels = data_frame['word'].to_list()
    data_frame.pop('word')
    eeg_data = data_frame.to_numpy()
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
    #return eeg_data[0:1000], word_labels[0:1000]
    return eeg_data, word_labels


def chunk_labels(labels, max_batch_size):
    is_par = len(labels) % max_batch_size
    num_batchs = len(labels)//max_batch_size if is_par == 0 else (len(labels)//max_batch_size)+1
    return [labels[batch_index*max_batch_size:(batch_index+1)*max_batch_size] for batch_index in range(num_batchs)]

def get_data_loader(eeg_data, word_labels):
    # Convertir labels a embeddings usando gpt2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    word_embeddings = []
    batched_labels = chunk_labels(word_labels, 1000)
    for batch in batched_labels:
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

        input_ids = inputs['input_ids']

        # Forward pass to get embeddings
        with torch.no_grad():
            outputs = model(input_ids)

        last_hidden_states = outputs.last_hidden_state
        #word_embeddings = torch.mean(last_hidden_states, dim=1)
        batch_embeddings = torch.mean(last_hidden_states, dim=1)
        word_embeddings.append(batch_embeddings)
        #print("Proccesed batch")

    word_embeddings = torch.cat(word_embeddings, dim=0)
    # Crear el dataset y dataloader
    dataset = EEGTextDataset(eeg_data, word_embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #return dataloader, word_embeddings
    return dataloader[:1000], word_embeddings[:1000]


# Definir un modelo para convertir datos de EEG a embeddings
class EEGToEmbeddingModel(nn.Module):
    def __init__(self, eeg_dim, embedding_dim):
        super(EEGToEmbeddingModel, self).__init__()
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, eeg_input):
        return self.eeg_encoder(eeg_input)


def train_model_to_embedding(dataloader, labels, eeg_data, word_embeddings, epochs):
    eeg_dim = eeg_data.shape[1]
    embedding_dim = word_embeddings.shape[1]
    eeg_model = EEGToEmbeddingModel(eeg_dim, embedding_dim)

    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(eeg_model.parameters(), lr=1e-4)

    # Entrenar el modelo de EEG a embedding
    eeg_model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            eeg_input, target_embeddings = batch
            optimizer.zero_grad()
            eeg_embeddings = eeg_model(eeg_input)
            loss = criterion(eeg_embeddings, target_embeddings)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return eeg_model

def get_file_names(folderpath : str) -> list:
    return os.listdir(folderpath)

def load_all_data(path):
    names = get_file_names(path)

    res_eeg_data = []
    res_labels = []

    for name in names:
        eeg_data, labels = load_data(path + name)    
        res_eeg_data = eeg_data if len(res_eeg_data) == 0 else np.append(res_eeg_data, eeg_data,0)
        res_labels.extend(labels)

    #res_eeg_data = torch.tensor(res_eeg_data, dtype=torch.float32)
    #res_eeg_data = torch.tensor(res_eeg_data)

    return res_eeg_data, res_labels

if __name__ == '__main__':
    #eeg_data, labels = load_data("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/resultsYAC_NR.csv")
    eeg_data, labels = load_all_data("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/")
    dataloader, word_embeddings = get_data_loader(eeg_data, labels)
    model = train_model_to_embedding(
        dataloader, 
        labels, 
        eeg_data, 
        word_embeddings,
        101
    )
    # torch.save(model, 'eeg_2_embeddings_model.pth')
    # # model_file = open("eeg_2_embeddings_model", 'wb')
    # # pickle.dump(model, model_file)
    # # model_file.close()

    # word_file = open("word_embeddings", 'wb')
    # pickle.dump(word_embeddings, word_file)
    # word_file.close()