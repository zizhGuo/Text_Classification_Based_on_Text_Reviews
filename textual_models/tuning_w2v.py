import multiprocessing
import gensim
import gensim.utils
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Word2Vec


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

def tuning_w2v_sizes(X):
    """ 
    This function tunes sizes (word embeddings dimension) for both SG and CBOW model
        Params:
            @X: the tokenized samples
        Return:
            void  
    """
    from sklearn.model_selection import cross_validate
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression

    cutWords_list = [x.split(" ") for x in X]
    count = 0

    # CBOW: sg == 0; SG: sg == 1
    for sg in [0, 1]:

        # word embeddings dimension
        for size in [10,20,30,40,50,100,150,200,250,300,400,500]:
            print([sg, size])
            start = datetime.datetime.now()
            word2vec_model = Word2Vec(cutWords_list, sg = sg, size=size, iter=30, min_count=5, workers=multiprocessing.cpu_count())
            
            # new samples
            X_new = [get_contentVector(cutWords, word2vec_model) for cutWords in cutWords_list]
 

 def tuning_w2v_1(X):
    """ 
    This function tunes iteration times for both SG and CBOW model
        Params:
            @X: the tokenized samples
        Return:
            void  
    """
    from sklearn.model_selection import cross_validate
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression

    cutWords_list = [x.split(" ") for x in X]
    count = 0
    for sg in [0, 1]:
        for iters in [1,2,3,4,5,10,15,20,30,40,50]:
            print([sg, iters])
            word2vec_model = Word2Vec(cutWords_list, sg = sg, size=300, iter=iters, min_count=5, workers=multiprocessing.cpu_count())
            X_new = [get_contentVector(cutWords, word2vec_model) for cutWords in cutWords_list] 
            print ('W2V fitting time: '+ str(end-start))
            # scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
            # clf = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter = 1000))
            # scores = cross_validate(clf, X_new, y, cv=10, scoring=scoring, return_train_score=False)
            # print_scores(scores)  

def tuning_w2v_window(X, y):
    """ 
    This function tunes window size for both SG and CBOW model
        Params:
            @X: the tokenized samples
            @y: the target values
        Return:
            void  
    """
    from sklearn.model_selection import cross_validate
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression

    cutWords_list = [x.split(" ") for x in X]
    count = 0
    for sg in [0, 1]:
        # for iters in [1,2,3,4,5,10,15,20,30,40,50]:
        f1_test_scores = []
        for window in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            print([sg, window])
            start = datetime.datetime.now()
            word2vec_model = Word2Vec(cutWords_list, sg = sg, window = window, size=300, iter=30, min_count=5, workers=multiprocessing.cpu_count())
            X_new = [get_contentVector(cutWords, word2vec_model) for cutWords in cutWords_list] 
            end = datetime.datetime.now()
            print ('W2V fitting time: '+ str(end-start))
            # scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
            # clf = OneVsRestClassifier(LogisticRegression(random_state=0, max_iter = 1000))
            # scores = cross_validate(clf, X_new, y, cv=10, scoring=scoring, return_train_score=False)
            # print_scores(scores)
            f1_test_scores.append(tuning_LR(X_new, y, count))
            count += 1
        df = pd.DataFrame(f1_test_scores, columns = ['f1_test_scores_l1liblinear', 'f1_test_scores_l2liblinear', 'f1_test_scores_l2newton-cg', 'f1_test_scores_l2lbfgs', 'f1_test_scores_l2sag'])
        df.to_csv('scores_test_w2v_window_sg'+str(sg)+'.csv')   