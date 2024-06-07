import os
import numpy as np
import h5py
import helpers.data_loading_helpers as dh

task = "TSR"

rootdir = "/Volumes/methlab/NLP/Ce_ETH/2019/FirstLevel_V2/"


sentences = {}
f = h5py.File("/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/task1-NR/Matlabfiles/resultsYAC_NR.mat", 'r')
#f = h5py.File("/home/mario/Desktop/zuco-dataset/task1-NR/Preprocessed/YAC/bip_YAC_NR5_EEG.mat")
#import scipy.io
#f = scipy.io.loadmat('task1-NR/Rawdata/YDG/YDG_NR1_EEG.mat')

sentence_data = f['sentenceData']
rawData = sentence_data['rawData']
contentData = sentence_data['content']
omissionR = sentence_data['omissionRate']
wordData = sentence_data['word']

# number of sentences:
# print(len(rawData))

words = {}
print("len(rawData)", len(rawData))
for idx in range(len(rawData)):
    obj_reference_content = contentData[idx][0]
    sent = dh.load_matlab_string(f[obj_reference_content])

    # get omission rate
    obj_reference_omr = omissionR[idx][0]
    omr = np.array(f[obj_reference_omr])
    #print(omr)

    # get word level data
    word_data = dh.extract_word_level_data(f, f[wordData[idx][0]])

    # number of tokens
    #print("len(word_data)", len(word_data))
    #print("-------------------")

    for widx in range(len(word_data)):

        # get first fixation duration (FFD)
        #print(word_data[widx]['word_idx'], ": ", word_data[widx]['content'], word_data[widx]['content1'])

        # get aggregated EEG alpha features
        #print(word_data[widx]['content'])
        word_string = word_data[widx]['content']
        words[word_string] = word_data[widx]
    #print(sentences)

print(words)


alpha_header = ["ALPHA_EEG_" + str(i) for i in range(1224)]
beta_header = ["BETA_EEG" + str(i) for i in range(1224)]
gamma_header = ["GAMMA_EEG" + str(i) for i in range(1224)]
theta_header = ["THETA_EEG" + str(i) for i in range(1224)]
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

rows = []
rows.append(header)

for word in words.keys():
    row = []
    row.extend(words[word]["ALPHA_EEG"])
    row.extend(words[word]["BETA_EEG"])
    row.extend(words[word]["GAMMA_EEG"])
    row.extend(words[word]["THETA_EEG"])
    row.append(words[word]["FFD"])
    row.append(words[word]["GD"])
    row.append(words[word]["GPT"])
    row.append(words[word]["TRT"])
    row.append(words[word]["nFix"])
    row.append(words[word]["word_idx"])
    row.append(words[word]["content"])
    rows.append(row)

print(rows)
