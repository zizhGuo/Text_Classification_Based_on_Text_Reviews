import numpy as np
import pandas as pd

import gensim
import gensim.utils
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec

import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer, SnowballStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

def lemmatizing_stemming(text):  
    """
    Words stemming or lemmatization: either to comment one of two return lines
        Params:
            @text: a string
        Return:
            a lematized or stemed string 
        
                
    """
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))   # stemming
    # return WordNetLemmatizer().lemmatize(text, pos='v')               # lemmatizing

def preprocess(text):
    """
    Remove Stopwords and stem/lematize
        Params:
            @text: a string
        Return:
            preprocessed string
    """
    result = ''
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            # toke = token.replace("!",'').replace("\\n",'').replace(")",'').replace("(",'').replace("!",'')
            # result = result + lemmatizing_stemming(toke).strip() + " "
            # result = result + re.sub("(?<![a-z])u(?![a-z])",'',line) + " "
            result= result + lemmatizing_stemming(re.sub("(?<![a-z])u(?![a-z])",'',token)).strip() + ' '
            # result= result + token + ' '
    return result