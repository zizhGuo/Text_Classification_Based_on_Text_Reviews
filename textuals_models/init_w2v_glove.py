import numpy as np
import pandas as pd

'''
@X: Tokenized samples

__________________________________________________________________
index	Stemmed Text 	            Lemmatized Text
__________________________________________________________________
0	    husband live nearbi stop	husband live nearby stop
1	    great attent servic great	great attentive service great
2	    locat great fresh brew	    location great fresh brew 
3	    nice restaur park vega	    nice restaurant park vegas
4	    star sushi joint serv size	make star sushi joint serve
__________________________________________________________________


'''


'''
@y: the corresponding target values
            
______________________________________________      
index   Multiclass Multilabel(one-hot encoded)
______________________________________________ 
0	    -1	        1	0	0
1	    1	        0	0	1
2	    1	        0	0	1
3	    0	        0	1	0
4	    1	        0	0	1
______________________________________________ 
'''

class Dataset():
    def __init__(self, path, **kwargs):
        print('Starting initialization')

        self.df = pd.read_csv(path)
        self.X = self.df['text']
        print(self.X[:10])
        # self.X_hours = get_regular_features(self.df)
        # print(self.X_hours[:10])
        cutWords_list = [x.split(" ") for x in self.X]

        from sklearn.preprocessing import LabelEncoder
        labelEncoder = LabelEncoder()
        self.y = labelEncoder.fit_transform(self.df['category'])
        print(self.y[:10])

        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse=False)
        self.y_one_hot = encoder.fit_transform(self.y.reshape((6000, 1)))

        for arg, value in kwargs.items():
            if arg == 'glove' and value == 1:
                self._isGloVe = True
                # _ = glove2word2vec('vectors.txt', 'test_glove.txt')
                glove_model = KeyedVectors.load_word2vec_format('test_glove.txt')
                self.X_glove = [get_contentVector(cutWords, glove_model) for cutWords in cutWords_list]
            elif arg == 'sg'and value == 1:
                self._isSG = True
                # train SG and save word vectors as Binary File
                # sg_model = Word2Vec(cutWords_list, sg = 1, size=300, iter=30, min_count=5, workers=multiprocessing.cpu_count())
                # sg_model.wv.save_word2vec_format("sg_wv.bin", binary=True)
                # Load Binary SG word vectors
                wv_sg = KeyedVectors.load_word2vec_format("sg_wv.bin", binary=True)
                self.X_sg = [get_contentVector(cutWords, wv_sg) for cutWords in cutWords_list]
            elif arg == 'cbow' and value == 1:
                self._isCBOW = True
                # cbow_model = Word2Vec(cutWords_list, sg = 0, size=300, iter=30, min_count=5, workers=multiprocessing.cpu_count())
                # cbow_model.wv.save_word2vec_format("cbow_wv.bin", binary=True)
                wv_cbow = KeyedVectors.load_word2vec_format("cbow_wv.bin", binary=True)
                self.X_cbow = [get_contentVector(cutWords, wv_cbow) for cutWords in cutWords_list]
        
        print('Initialization finished')

    def get_X_SG(self):
        print(self.X_sg[:1])
        return self.X_sg

    def get_X_CBOW(self):
        print(self.X_cbow[:1])
        return self.X_cbow

    def get_X_GloVe(self):
        print(self.X_glove[:1])
        return self.X_glove

    def get_y(self):
        return self.y