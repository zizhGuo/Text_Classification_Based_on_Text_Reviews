import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def tf_idf(X):
    """ 
    This function ceates the tf-idf model based on the tokenized documents
        Params:
            @X: a list of list/dataframe documents
        Return:
            void  
    """
    print(X.shape)
    cv=CountVectorizer() 
    word_count_vector=cv.fit_transform(X)
    print(word_count_vector.shape)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector)

    # the extracted tf-idf vectors in 300 dimensions
    X = tfidf_transformer.transform(word_count_vector)