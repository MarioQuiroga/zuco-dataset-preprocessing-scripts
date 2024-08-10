import torch
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, GPT2Model
from utils.EEGTextDataset import *
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re


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
    #return eeg_data[0:100], word_labels[0:100]
    return eeg_data, word_labels

def chunk_labels(labels, max_batch_size):
    is_par = len(labels) % max_batch_size
    num_batchs = len(labels)//max_batch_size if is_par == 0 else (len(labels)//max_batch_size)+1
    return [labels[batch_index*max_batch_size:(batch_index+1)*max_batch_size] for batch_index in range(num_batchs)]

def get_embeddings(word_labels):
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
        batch_embeddings = torch.mean(last_hidden_states, dim=1)
        word_embeddings.append(batch_embeddings)
        #print("Proccesed batch")

    word_embeddings = torch.cat(word_embeddings, dim=0)
    return word_embeddings

def get_data_loader(eeg_data, word_labels, word_embeddings):
    # Convertir labels a embeddings usando gpt2
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Crear el dataset y dataloader
    dataset = EEGTextDataset(eeg_data, word_embeddings)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    #return dataloader, word_embeddings
    return dataloader

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

    res_eeg_data = torch.tensor(res_eeg_data, dtype=torch.float32)
    #res_eeg_data = torch.tensor(res_eeg_data)

    return res_eeg_data, res_labels
    #return res_eeg_data[:1000], res_labels[:1000]


def train_test_split_eeg_data(data, labels):
    x_train, y_train, x_test, y_test = [], [], [], [] 
    # Guardar en dict donde key es la palabra y el value es la lista de ejemplos eeg data
    word_dict = {}
    stop_words = get_stop_words()
    for i in range(len(labels)):
        cleaned_word = clean_word(labels[i])
        if (not cleaned_word in stop_words) and valid_word(cleaned_word):    
            keys = word_dict.keys()
            if cleaned_word in keys:
                word_dict[cleaned_word].append(data[i])
            else:
                word_dict[cleaned_word] = [data[i]]

    # Por cada palabra splitear la lista de ejemplos y guardar en 2 vectores 
    for word in word_dict.keys():
        X = word_dict[word]
        if (len(X) > 10): # Teniendo en cuenta que la media es 16
            y = [word for i in range(len(X))]
            x_train_word, x_test_word, y_train_word, y_test_word = train_test_split(X, y, test_size=0.20, random_state=42)
            if len(x_train) == 0:
                x_train = x_train_word
                y_train = y_train_word
                x_test = x_test_word
                y_test = y_test_word
            else:
                x_train.extend(x_train_word)
                y_train.extend(y_train_word)
                x_test.extend(x_test_word)
                y_test.extend(y_test_word)

    #x_train, y_train = shuffle(x_train, y_train)
    #x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_test, y_test

def get_unique_labels_embedddings(labels, word_embeddings):
    unique_labels_embeddings, unique_labels = [], []
    for i in range(len(labels)):
        if labels[i] not in unique_labels:
            unique_labels_embeddings.append(word_embeddings[i])
            unique_labels.append(labels[i])
    return unique_labels_embeddings, unique_labels

def get_stop_words():
    return ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "III", "mr", "0-3", "828-447", "II", "p-38", "us"]

def clean_word(word):
    word_res = word.replace('"', '').replace('[', "").replace(']', "").replace('(', "").replace(')', "").replace("'", "").replace('%', "").replace(',', "").replace('.', "").replace(':', "").replace('!', "").replace('$', "").replace('?', "").replace('¿', "").replace('¡', "")
    if word_res[0] == '-':
        word_res = word_res[1:]
    if word_res[len(word_res)-1] == '-':
        word_res = word_res[:-1]
    return word_res.lower()
    
def valid_word(word):
    return len(word) > 2 and word != 'nan' and not word.isdigit() and not word[:len(word)-1].isdigit() and not re.search("\d\d\d\d-\d\d-\d\d", word) and not re.search("\d\d\d\d-\d\d\d\d", word) and not re.search(".\d\d\d\d-\d\d\d\d", word)