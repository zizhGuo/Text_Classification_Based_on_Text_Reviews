import numpy as np
import pandas as pd
import sys
import csv
import json
import os
import matplotlib.pyplot as plt
import re
import string

import gensim
import gensim.utils
from gensim.parsing.preprocessing import STOPWORDS
# from nltk.stem import WordNetLemmatizer, SnowballStemmer
# import nltk

import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def converter_json2csv(json_file_path, csv_file_path):
    """ 
    This function reads a JSON file and writes it in CSV file
        Params:
            @json_file_path: a string represents the json file path
            @csv_file_path: a string represents the output csv file path
        Return:
            void  
    """
    with open(json_file_path,'r',encoding='utf-8') as fin:
        for line in fin:
            line_contents = json.loads(line)
            break
        # print(headers)
    with open(csv_file_path, 'w', newline='',encoding='utf-8') as fout:
        writer=csv.DictWriter(fout, headers)
        writer.writeheader()
        with open(json_file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_contents = json.loads(line)
                writer.writerow(line_contents)

def write_filtered_csv(csv_file_path, btable):
    ''' 
    This function picks equal numbers of three classes,
        and select three features: hours, stars, text;
        Append them into a single dataframe;
        Write it in c.csv file.
        Params:
            @csv_file_path: a string represents the input CSV file
            @btable: an indexed dataframe
        Return:
            void  
    '''
    class_f_size = 2000 # record number for each category
    class_a_size = 2000
    class_s_size = 2000
    chunk_id = 0
    indices = btable.index # the string representing the business_id as the indices
    
    # create a new dataframe containing filtered records
    df = pd.DataFrame(columns= ['hours', 'stars', 'text', 'category'])

    # loop over each chunks to filter wanted record
    for chunk in pd.read_csv(csv_file_path, chunksize= 100000):
        if class_f_size <= 0 and class_a_size <= 0 and class_s_size <= 0:
            break
        print(chunk_id)
        print(datetime.datetime.now())
        chunk_id += 1
        
        # each line in one chunk
        for i in range(0, len(chunk)):
            business_id = chunk.iloc[i, :]['business_id']
            if business_id in indices:
                restaurant = btable.loc[business_id]
                hours = restaurant['hours']
                cat = restaurant['categories']
                stars = chunk.iloc[i, :]['stars']
                text = chunk.iloc[i, :]['text']
                if class_f_size > 0 and 'Fast Food' in cat and 'Sushi Bars' not in cat and 'American (New)' not in cat:
                    df.loc[len(df)] = [hours, stars, text, 'Fast Food']
                    class_f_size -= 1
                if class_s_size > 0 and 'Fast Food' not in cat and 'Sushi Bars' in cat and 'American (New)' not in cat:
                    df.loc[len(df)] = [hours, stars, text, 'Sushi Bars']
                    class_s_size -= 1
                if class_a_size > 0 and 'Fast Food' not in cat and 'Sushi Bars' not in cat and 'American (New)' in cat:
                    df.loc[len(df)] = [hours, stars, text, 'American (New)']
                    class_a_size -= 1
    
    # write the new dataset to a new CSV file
    df.to_csv('c.csv', encoding='utf-8', header = True)

