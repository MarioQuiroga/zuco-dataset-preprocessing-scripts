import pandas as pd
import matplotlib.pyplot as plt
from helpers.read_matlab_files_to_csv import *
import numpy as np
from scipy import stats
import matplotlib
plt.style.use('ggplot')
matplotlib.use('tkagg')

input_path = "/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/"

if __name__ == '__main__':
    file_names = get_file_names(input_path)
    words_count = {}

    max_words = 0
    for file in file_names:
        df = pd.read_csv(input_path + file)
        words = df["word"]
        for word in words:
            max_words += 1
            if words_count.get(word) is None:
                words_count[word] = 1
            else:
                words_count[word] = words_count[word] + 1

    print(f'Total words: {max_words}')

    unique_words = len(words_count.keys())
    print(f'Unique words: {unique_words}')
    
    #average_word_count = sum(words_count.values()) / unique_words
    #print(f'Average word count: {average_word_count}')

    print(f'Min word count: {np.min(list(words_count.values()))}')

    print(f'Max word count: {np.max(list(words_count.values()))}')

    print(f'Mean word count: {np.mean(list(words_count.values()))}')

    print(f'Median word count: {np.median(list(words_count.values()))}')

    print(f'Mode word count: {stats.mode(list(words_count.values()))}')


    # counts, bins = np.histogram(x)
    # plt.stairs(counts, bins)
    #plt.hist(list(words_count.values()))
    

    rows = [["word", "count"]]
    for word in words_count.keys():
        clean_word = str(word).replace('"', '').replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace("'", '').replace('()', '')
        rows.append([clean_word, words_count[word]])

    counts, bins = np.histogram([r[1] for r in rows[1:]])
    plt.stairs(counts, bins)
    plt.show()
    #saveCSVFile("cantidades.csv", rows)

    





