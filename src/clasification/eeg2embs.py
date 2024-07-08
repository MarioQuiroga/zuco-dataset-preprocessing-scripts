import torch
import torch.nn as nn
from utils.preprocessing import *
from utils.EEGToEmbeddingModel import *
from utils.EEGTextDataset import *
import pickle
import os

csvs_path = "/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/"

if __name__ == '__main__':
    #eeg_data, labels = load_data("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/resultsYAC_NR.csv")
    print("Get data")
    eeg_data, labels = load_all_data(csvs_path)
    x_train, y_train, x_test, y_test = train_test_split_eeg_data(eeg_data, labels)
    dataloader, word_embeddings = get_data_loader(x_train, y_train)
    print("Training...")
    model = train_model_to_embedding(
        dataloader, 
        eeg_data, 
        word_embeddings,
        2
    )

    print("Testing...")
    dataloader_test, word_embeddings_test = get_data_loader(x_test, y_test)
    unique_labels_embeddings, unique_labels = get_unique_labels_embedddings(y_test, word_embeddings_test)

    eval_model_as_classifier(model, x_test, y_test, unique_labels_embeddings, unique_labels)

    # torch.save(model, 'eeg_2_embeddings_model.pth')
    # # model_file = open("eeg_2_embeddings_model", 'wb')
    # # pickle.dump(model, model_file)
    # # model_file.close()

    # word_file = open("word_embeddings", 'wb')
    # pickle.dump(word_embeddings, word_file)
    # word_file.close()