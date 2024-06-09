import os
import numpy as np
import h5py
import helpers.data_loading_helpers as dh
import csv

def getWordLevelData(f):
    sentence_data = f['sentenceData']
    rawData = sentence_data['rawData']
    contentData = sentence_data['content']
    omissionR = sentence_data['omissionRate']
    wordData = sentence_data['word']
    # number of sentences:
    # print(len(rawData))
    words = []
    for idx in range(len(rawData)):
        obj_reference_content = contentData[idx][0]
        sent = dh.load_matlab_string(f[obj_reference_content])
        # get word level data
        word_data = dh.extract_word_level_data(f, f[wordData[idx][0]])

        for widx in range(len(word_data)):
            #print(word_data[widx]['word_idx'], ": ", word_data[widx]['content'], word_data[widx]['content1'])
            word_string = word_data[widx]['content']
            word_data_row = [word_string]
            word_data_row.append(word_data[widx])
            words.append(word_data_row)
    return words


def get_headers_word_level_data_to_rows():
    alpha_header = ["ALPHA_EEG_" + str(i) for i in range(306)]
    beta_header = ["BETA_EEG" + str(i) for i in range(306)]
    gamma_header = ["GAMMA_EEG" + str(i) for i in range(306)]
    theta_header = ["THETA_EEG" + str(i) for i in range(306)]
    header = []
    header.extend(alpha_header)
    header.extend(beta_header)
    header.extend(gamma_header)
    header.extend(theta_header)
    header.append("FFD")
    header.append("GD")
    header.append("GPT")
    header.append("TRT")
    header.append("nFix")
    header.append("word_idx")
    header.append("word")
    return header

def word_level_data_to_rows(words):
    header = get_headers_word_level_data_to_rows()
    rows = []
    rows.append(header)
    for idx in range(len(words)):
        word = words[idx][0]
        word_data = words[idx][1]
        row = []
        row.extend(word_data["ALPHA_EEG"])
        row.extend(word_data["BETA_EEG"])
        row.extend(word_data["GAMMA_EEG"])
        row.extend(word_data["THETA_EEG"])
        row.append(word_data["FFD"])
        row.append(word_data["GD"])
        row.append(word_data["GPT"])
        row.append(word_data["TRT"])
        row.append(word_data["nFix"])
        row.append(word_data["word_idx"])
        row.append(word_data["content"])
        if len(row) == 1231:
            rows.append(row)

    return rows

def saveCSVFile(filename, content):
    with open(filename, 'w') as f:
        # Create a CSV writer object that will write to the file 'f'
        csv_writer = csv.writer(f)
        
        # Write the field names (column headers) to the first row of the CSV file
        csv_writer.writerows(content)

def get_file_names(filename):
    return os.listdir(filename)

if __name__ == "__main__":
    f = h5py.File("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/task1-NR/Matlabfiles/resultsYAC_NR.mat", 'r')
    words = getWordLevelData(f)
    rows = word_level_data_to_rows(words)
    saveCSVFile("mat_lab_file.csv", rows)
