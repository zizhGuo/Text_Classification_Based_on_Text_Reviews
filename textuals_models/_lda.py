import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation

def lda(X):
    """ 
    This function ceates the LDA model based on the tokenized documents
        Params:
            @X: a list of list/dataframe documents
        Return:
            void  
    """
    X = X.apply(preprocess)
    cv=CountVectorizer() 
    word_count_vector=cv.fit_transform(X)
    from sklearn.decomposition import LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=300, random_state=0)
    
    # the extracted LDA topics vectors in 300 dimensions
    X_topics = lda.fit_transform(word_count_vector)
