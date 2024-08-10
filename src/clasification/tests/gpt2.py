import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # For warning: oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Model
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing

# Dataset personalizado
class EEGTextDataset(Dataset):
    def __init__(self, eeg_data, token_labels):
        self.eeg_data = eeg_data
        self.token_labels = token_labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.token_labels[idx]

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)

# Modelo personalizado que acepta vectores EEG como entrada
class EEG2TextModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(EEG2TextModel, self).__init__()
        self.eeg_encoder = nn.Linear(hidden_dim, output_dim)
        self.transformer = GPT2LMHeadModel.from_pretrained('gpt2')

    def forward(self, eeg_input, labels=None):
        eeg_encoded = self.eeg_encoder(eeg_input)
        outputs = self.transformer(inputs_embeds=eeg_encoded, labels=labels)
        return outputs.loss, outputs.logits

def get_data_loader(eeg_data, word_labels):
    # Tokenización de palabras
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #token_labels = tokenizer(word_labels, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True)
    inputs = tokenizer(word_labels, return_tensors='pt', padding=True, truncation=True, is_split_into_words=True)

    # Obtener los embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # La última capa oculta (hidden state)
    last_hidden_states = outputs.last_hidden_state

    # Extraer los embeddings de las palabras (promedio de los tokens para cada palabra)
    token_labels = torch.mean(last_hidden_states, dim=1)
    #.encode(word_labels)
    #token_labels = [tokenizer.encode(word) for word in word_labels]
    #token_labels = torch.tensor(token_labels)

    eeg_data_filter = eeg_data[:1000]
    token_labels = token_labels[:1000]

    dataset = EEGTextDataset(eeg_data_filter, token_labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader

def load_data(filename):
    data_frame = pd.read_csv(filename)
    word_labels = data_frame['word'].to_list()
    data_frame.pop('word')
    eeg_data = data_frame.to_numpy()
    # mean = eeg_data.mean(axis=1)
    # std = eeg_data.std(axis=1)
    # eeg_data = [(data - mean) /std for data in eeg_data]
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
    return eeg_data, word_labels

def train_model(model, dataloader, optimizer):
    model.apply(init_weights)
    model.train()
    for epoch in range(10):
        for batch in dataloader:
            eeg_input, labels = batch
            optimizer.zero_grad()
            loss, logits = model(eeg_input, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')


if __name__ == '__main__':
    eeg_data, word_labels = load_data("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/resultsYAC_NR.csv")
    dataloader = get_data_loader(eeg_data, word_labels)

    # Parámetros del modelo
    hidden_dim = eeg_data.shape[1]
    output_dim = 768  # Tamaño de los embeddings de GPT-2

    # Entrenamiento del modelo
    model = EEG2TextModel(hidden_dim, output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.001)
    train_model(model, dataloader, optimizer)