import numpy as np
import pandas as pd

class LR():
    """ 
    This class encapsulate an LR classifier, the sameples X 
    and the target values y.

    It implements Sklearn Library Logistic Regression model 
    and it can fit the model with given training data.
    """
    def __init__(self, X, y):
        # super(LR, self).__init__(random_state=0, max_iter = 1000)
        from sklearn.linear_model import LogisticRegression
        self.clf = LogisticRegression(random_state=0, max_iter = 1000)
        self.X = X
        self.y = y
    
    def fit_transform(self):
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        param_grid = [
            {'penalty': ['l2'], 'solver': ['newton-cg']},
            ]
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.clf, param_grid, cv=10,
                          scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(self.X, self.y)
        print(grid_search.cv_results_['mean_fit_time'])
        print(grid_search.cv_results_['param_solver'])
        print(grid_search.cv_results_['rank_test_f1_macro'])
        print(grid_search.cv_results_['mean_test_f1_macro'])
        print(grid_search.cv_results_['mean_test_accuracy'])
        print('--------------------------------------------------')
        print('Tune LR ends:')
        return grid_search.cv_results_['mean_test_f1_macro']

class SVM():
    """ 
    This class encapsulate an SVC (Support Vector Classification) classifier, 
    the sameples X and the target values y.
    
    It implements Sklearn Library SVC model 
    and it can fit the model with given training data.
    """
    def __init__(self, X, y):
        # super(LR, self).__init__(random_state=0, max_iter = 1000)
        from sklearn.svm import SVC
        from sklearn.multiclass import OneVsOneClassifier
        self.clf = OneVsOneClassifier(SVC())
        self.X = X
        self.y = y
    
    def fit_transform(self):
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        param_grid = [
            {'estimator__kernel': ['rbf'], "estimator__C":[10], "estimator__gamma":[1.0]},
            ]
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.clf, param_grid, cv=10,
                          scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(self.X, self.y)
        print(grid_search.cv_results_['mean_fit_time'])
        # print(grid_search.cv_results_['param_solver'])
        # print(grid_search.cv_results_['rank_test_f1_macro'])
        # print(grid_search.cv_results_['mean_test_f1_macro'])
        print(grid_search.cv_results_['mean_test_accuracy'])
        print('--------------------------------------------------')
        print('Tune SVM ends:')
        return grid_search.cv_results_['mean_test_f1_macro']

class MNB():
    """ 
    This class encapsulate an MNB classifier, the sameples X 
    and the target values y.

    It implements Sklearn Library multinomial Naive Bayes model 
    and it can fit the model with given training data.
    """
    def __init__(self, X, y):
        # super(LR, self).__init__(random_state=0, max_iter = 1000)
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.preprocessing import MinMaxScaler
        self.clf = MultinomialNB()
        self.X = MinMaxScaler().fit_transform(X)
        self.y = y
    
    def fit_transform(self):
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        param_grid = [
            {'alpha': [1.0], 'fit_prior':[True]},
            ]
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.clf, param_grid, cv=10,
                          scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(self.X, self.y)
        print(grid_search.cv_results_['mean_fit_time'])
        # print(grid_search.cv_results_['param_solver'])
        # print(grid_search.cv_results_['rank_test_f1_macro'])
        # print(grid_search.cv_results_['mean_test_f1_macro'])
        print(grid_search.cv_results_['mean_test_accuracy'])
        print('--------------------------------------------------')
        print('Tune MNB ends:')
        return grid_search.cv_results_['mean_test_f1_macro']

class RF():
    """ 
    This class encapsulate an RF classifier, the sameples X 
    and the target values y.

    It implements Sklearn Library Random Forest model 
    and it can fit the model with given training data.
    """
    def __init__(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import MinMaxScaler
        self.clf = RandomForestClassifier()
        self.X = MinMaxScaler().fit_transform(X)
        self.y = y
    
    def fit_transform(self):
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        param_grid = [
            {'max_depth': [1000]}
            ]
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.clf, param_grid, cv=10,
                          scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(self.X, self.y)
        print(grid_search.cv_results_['mean_fit_time'])
        # print(grid_search.cv_results_['param_solver'])
        # print(grid_search.cv_results_['rank_test_f1_macro'])
        # print(grid_search.cv_results_['mean_test_f1_macro'])
        print(grid_search.cv_results_['mean_test_accuracy'])
        print('--------------------------------------------------')
        print('Tune RF ends:')
        return grid_search.cv_results_['mean_test_f1_macro']

class NN():
    """ 
    This class encapsulate an k-NN classifier, the sameples X 
    and the target values y.

    It implements Sklearn Library k nearest neighbors model 
    and it can fit the model with given training data.
    """
    def __init__(self, X, y):
        from sklearn.neighbors import KNeighborsClassifier
        self.clf = KNeighborsClassifier()
        self.X = X
        self.y = y
    
    def fit_transform(self):
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        param_grid = [
            {'n_neighbors': [7]}
            ]
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(self.clf, param_grid, cv=10,
                          scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(self.X, self.y)
        print(grid_search.cv_results_['mean_fit_time'])
        # print(grid_search.cv_results_['param_solver'])
        # print(grid_search.cv_results_['rank_test_f1_macro'])
        # print(grid_search.cv_results_['mean_test_f1_macro'])
        print(grid_search.cv_results_['mean_test_accuracy'])
        print('--------------------------------------------------')
        print('Tune NN ends:')
        return grid_search.cv_results_['mean_test_f1_macro']

def tuning_Perceptron_tol(X, y):
    """ 
    This function works for tuning perceptron learning model on stopping criteria
        using sklearn library.
        Params:
            @X: a sameple dataframes of sequences embeddings
            @y: a list of multiclassed label
        Return:
            void  
    """
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Perceptron
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    param_grid = [
                {'penalty': ['l2'],'alpha': [0.0001],'max_iter':[1000]}
                ]
    f1_test_scores = []
    for tol in np.arange(1.0, 4.1, 0.1):
        _tol = 10 **(-tol) 
        clf = Perceptron(random_state=0, n_iter_no_change = 5, tol = _tol)
        grid_search = GridSearchCV(clf, param_grid, cv=10,
                                scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(X, y)
        # df_results = pd.DataFrame.from_dict(grid_search.cv_results_)
        f1_test_scores.append(grid_search.cv_results_['mean_test_f1_macro'][0])
    print(f1_test_scores)
    # df = pd.DataFrame(f1_test_scores, columns = ['perceptron_tol'])
    df = pd.read_csv('perceptron_tol.csv')
    # df['tol'] = np.arange(1.0, 4.1, 0.1)
    df['perceptron_tol_l2'] = f1_test_scores
    df.to_csv('perceptron_tol.csv')
    return grid_search.cv_results_['mean_test_f1_macro']  

def tuning_Perceptron_alpha(X, y):
    """ 
    This function works for tuning perceptron learning model on alpha
        using sklearn library.
        :alpha the constant regularization
        Params:
            @X: a sameple dataframes of sequences embeddings
            @y: a list of multiclassed label
        Return:
            void  
    """
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Perceptron
    scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
    param_grid = [
                {'penalty': ['l2'],'tol':[1e-4],'max_iter':[1000]}
                ]
    f1_test_scores = []
    for alpha in np.arange(2.0, 5.1, 0.1):
        _alpha = 10 **(-alpha) 
        clf = Perceptron(random_state=0, n_iter_no_change = 5, alpha = _alpha)
        grid_search = GridSearchCV(clf, param_grid, cv=10,
                                scoring=scoring, return_train_score = True, refit=False, iid = True)
        grid_search.fit(X, y)
        # df_results = pd.DataFrame.from_dict(grid_search.cv_results_)
        f1_test_scores.append(grid_search.cv_results_['mean_test_f1_macro'][0])
    print(f1_test_scores)
    # df = pd.DataFrame(f1_test_scores, columns = ['perceptron_alpha'])
    df = pd.read_csv('perceptron_alpha.csv')
    df['perceptron_alpha_l2'] = f1_test_scores
    # df['alpha'] = np.arange(1.0, 4.1, 0.1)
    df.to_csv('perceptron_alpha.csv')
    return grid_search.cv_results_['mean_test_f1_macro'] 

