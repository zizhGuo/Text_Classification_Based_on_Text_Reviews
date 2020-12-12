import numpy as np
import pandas as pd
import gensim

def get_contentVector(cutWords, word2vec_model):
    """ 
    This function converts the list of the tokenized sample into a list of word embeddings.
        :The sequence embeddings comes from the averaged words embeddings
        :The words not in the model would be replaced by a zero embedding 
            (for w2v model, in glove using <unk> token)
        Params:
            @cutWords: a list of tokenized word of one sample record
            @word2vec_model: a Gensim Library trained Word2Vec model
        Return:
            void  
    """
    vector_list = [word2vec_model.wv[k] for k in cutWords if k in word2vec_model]
    if not vector_list:
        contentVector = [0] * size
        print(cutWords)
        print('1 more')
    else:
        contentVector = np.array(vector_list).mean(axis=0)
    return contentVector

def tuning_LR(X, y, index):
    """ 
    This function works for tuning the LR model using Sklearn library grid search method.
        Params:
            @X: a sameple dataframes of sequences embeddings
            @y: a list of multiclassed label
            @index: the index for convinient naming storing the result file
        Return:
            void  
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    
    param_grid = [
                {'penalty': ['l1'], 'solver': ['liblinear']},
                {'penalty': ['l2'], 'solver': ['liblinear','newton-cg', 'lbfgs', 'sag']},
                ]
    # clf = OneVsRestClassifier(LogisticRegression(random_state=0))
    clf = LogisticRegression(random_state=0, max_iter = 1000)
    grid_search = GridSearchCV(clf, param_grid, cv=10,
                          scoring=scoring, return_train_score = True, refit=False, iid = True)
    grid_search.fit(X, y)
    df_results = pd.DataFrame.from_dict(grid_search.cv_results_)
    # df_results.to_csv('scores_test'+'_'+ str(index)+'.csv')
    df_results.to_csv('scores_test'+'_iter_'+ str(index)+'.csv')
    print(grid_search.cv_results_['mean_fit_time'])
    print(grid_search.cv_results_['param_solver'])
    print(grid_search.cv_results_['rank_test_f1_macro'])
    print(grid_search.cv_results_['mean_test_f1_macro'])
    print('--------------------------------------------------')
    return grid_search.cv_results_['mean_test_f1_macro']

def tune_glove_size(X, y):
    """ 
    This function works for tuning the sizes(word embeddings dimension) for GloVe
        :It uses word embeddings trained from the GloVe models with different sizes
        :The GloVe models were trained by Stanford GloVe program (in c++)
        Params:
            @X: a sameple dataframes of sequences embeddings
            @y: a list of multiclassed label
        Return:
            void  
    """
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.scripts.glove2word2vec import glove2word2vec
    sizes = [10,20,30,40,50,100,150,200,250,300,400]
    # sizes = [10,20,30]
    f1_test_scores = []
    for size in sizes:
        _ = glove2word2vec('vectors'+str(size)+'.txt', 'test_word2vec.txt')
        from gensim.models import KeyedVectors
        word2vec_model = KeyedVectors.load_word2vec_format('test_word2vec.txt')
        cutWords_list = [x.split(" ") for x in X]
        X_new = [get_contentVector(cutWords, word2vec_model, size) for cutWords in cutWords_list]
        # print(X_new[:10])
        f1_test_scores.append(tuning_LR(X_new, y, 0))
    print(f1_test_scores)
    df = pd.DataFrame(f1_test_scores, columns = ['f1_test_scores_l1liblinear', 'f1_test_scores_l2liblinear', 'f1_test_scores_l2newton-cg', 'f1_test_scores_l2lbfgs', 'f1_test_scores_l2sag'])
    df.to_csv('scores_test.csv')
    return f1_test_scores

def tune_glove_iter(X, y):
    """ 
    This function works for tuning the iteration times for GloVe
        :It uses word embeddings trained from the GloVe models with different sizes
        :The GloVe models were trained by Stanford GloVe program (in c++)
        Params:
            @X: a sameple dataframes of sequences embeddings
            @y: a list of multiclassed label
        Return:
            void  
    """
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.scripts.glove2word2vec import glove2word2vec
    iters = [1,2,3,4,5,10,15,20,30,40,50]
    # sizes = [10,20,30]
    f1_test_scores = []
    for iter in iters:
        _ = glove2word2vec('vectors_iter_'+str(iter)+'.txt', 'test_word2vec.txt')
        from gensim.models import KeyedVectors
        word2vec_model = KeyedVectors.load_word2vec_format('test_word2vec.txt')
        cutWords_list = [x.split(" ") for x in X]
        X_new = [get_contentVector(cutWords, word2vec_model, iter) for cutWords in cutWords_list]
        # print(X_new[:10])
        f1_test_scores.append(tuning_LR(X_new, y, 0))
    print(f1_test_scores)
    df = pd.DataFrame(f1_test_scores, columns = ['f1_test_scores_l1liblinear', 'f1_test_scores_l2liblinear', 'f1_test_scores_l2newton-cg', 'f1_test_scores_l2lbfgs', 'f1_test_scores_l2sag'])
    df.to_csv('scores_test_iters.csv')
    return f1_test_scores

def tune_glove_window(X, y):
    """ 
    This function works for tuning the window sizes for GloVe
        :It uses word embeddings trained from the GloVe models with different sizes
        :The GloVe models were trained by Stanford GloVe program (in c++)
        Params:
            @X: a sameple dataframes of sequences embeddings
            @y: a list of multiclassed label
        Return:
            void  
    """
    from gensim.test.utils import datapath, get_tmpfile
    from gensim.scripts.glove2word2vec import glove2word2vec
    windows = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # sizes = [10,20,30]
    f1_test_scores = []
    for window in windows:
        _ = glove2word2vec('vectors_window_'+str(window)+'.txt', 'test_word2vec.txt')
        from gensim.models import KeyedVectors
        word2vec_model = KeyedVectors.load_word2vec_format('test_word2vec.txt')
        cutWords_list = [x.split(" ") for x in X]
        X_new = [get_contentVector(cutWords, word2vec_model, window) for cutWords in cutWords_list]
        # print(X_new[:10])
        f1_test_scores.append(tuning_LR(X_new, y, 0))
    print(f1_test_scores)
    df = pd.DataFrame(f1_test_scores, columns = ['f1_test_scores_l1liblinear', 'f1_test_scores_l2liblinear', 'f1_test_scores_l2newton-cg', 'f1_test_scores_l2lbfgs', 'f1_test_scores_l2sag'])
    df.to_csv('scores_test_windows.csv')
    return f1_test_scores