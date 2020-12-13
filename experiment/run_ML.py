import numpy as np
import pandas as pd

# import init_w2v_glove.py

def main():
    start = datetime.datetime.now()
    
    dataset = Dataset('IS_new.csv', glove = 0, sg = 0, cbow = 0)
    dataset = Dataset('IS.csv', glove = 0, sg = 0, cbow = 0)
    # X_glove = dataset.get_X_GloVe()
    # X_sg = dataset.get_X_SG()
    # X_cbow = dataset.get_X_CBOW()
    y = dataset.get_y()

    # clf = LR(X_cbow, y)
    # print(clf.fit_transform())

    # clf2 = SVM(X_cbow, y)
    # print(clf2.fit_transform())

    # clf3 = MNB(X_cbow, y)
    # print(clf3.fit_transform())  

    # clf4 = RF(X_cbow, y)
    # print(clf4.fit_transform())    

    # clf5 = NN(X_cbow, y)
    # print(clf5.fit_transform()) 

    end = datetime.datetime.now()
    print (end-start)
if __name__ == "__main__":
    main()