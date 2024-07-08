from helpers.read_matlab_files_to_csv import *
import h5py

data_path = "/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/task1-NR/Matlabfiles/"
output_path = "/home/mario/Desktop/zuco-dataset-preprocessing-scripts/data/matfiles_csv/"

if __name__ == "__main__":
    names = get_file_names(data_path)
    for filename in names:
        print("Processing file: " + filename)
        f = h5py.File(data_path + filename, 'r')
        words = getWordLevelData(f)
        rows = word_level_data_to_rows(words)
        saveCSVFile(output_path + filename.split(".")[0] + ".csv", rows)
    print("Finished processing files")