import pickle
import os

def read_pickle(filename):
    with open(filename, "rb") as file:
        data = pickle.load(file)
    return data

def merge_files(output_path, parts):
    with open(output_path, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
