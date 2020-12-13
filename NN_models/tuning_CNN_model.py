import tensorflow as tf
import tensorflow.keras as keras
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import numpy as np
import pandas as pd
# import create_CNN_model

def tuning_CNN_trainable():
    """ 
    This function works for tuning the CNN model in static or non-static.
    """
    df = pd.read_csv('IS_new.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    X = df['text']

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['category'])
    # print(y[:10])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y = y.reshape((6000, 1))
    y_one_hot = encoder.fit_transform(y)
    # print(y_one_hot[:10])    

    # compute the maximum length of sequence
    # maxlen = 0
    # for sequence in [x[:80000].split(" ") for x in X]:
    #     if len(sequence) > maxlen:
    #         maxlen = len(sequence)
    # print('maximum maxlen: %s' % str(maxlen) )

    acc_trainable_size = []
    acc_val_trainable_size = []
    for trainable_size in [True, False]:
        from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1000)
        text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
        vectorizer.adapt(text_ds)
        # print(vectorizer.get_vocabulary()[:10])
        # print(len(vectorizer.get_vocabulary()))
        # x_ix = vectorizer(np.array([[s[:6000]] for s in X])).numpy()
        x_ix = vectorizer(X).numpy()
        # print(X[0])
        print(x_ix[:10])
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(x_ix, y_one_hot, test_size=0.33, random_state=42)
        # print(y_train[:5])
        from sklearn.model_selection import KFold
        acc_10_CV = []
        val_acc_10_CV = []
        for train_index, test_index in KFold(10).split(x_ix):
            X_train, X_test= x_ix[train_index], x_ix[test_index]
            y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
            model = create_model(vectorizer = vectorizer, maxlen = 1000, region_size = 3, trainable = trainable_size)
            history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
            history_dict = history.history
            acc_10_CV.append(history_dict['accuracy'][-1])
            val_acc_10_CV.append(history_dict['val_accuracy'][-1])
        mean_acc = np.mean(acc_10_CV)
        mean_acc_val = np.mean(val_acc_10_CV)
        print('trainable size: ' + str(trainable_size))
        print(mean_acc)
        print(mean_acc_val)
        acc_trainable_size.append(mean_acc)
        acc_val_trainable_size.append(mean_acc_val)
    print('trainable tune final results: ')
    print(acc_trainable_size)
    print(acc_val_trainable_size)
    df4 = pd.DataFrame(columns = ['acc_trainable_size', 'acc_val_trainable_size'])
    df4['acc_trainable_size'] = acc_trainable_size
    df4['acc_val_trainable_size'] = acc_val_trainable_size
    df4.to_csv('CNN_trainable.csv',  header = True)

    def main1():
    df = pd.read_csv('Queen.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    # df.to_csv('Queenasdwa.csv')
    # print(df.head(10))
    # print(df['business_id'].value_counts())
    # print(df['category'].value_counts())
    X = df['text']
    # print(type(X[0]))
    # print(X[0])
    # print(len(X[0]))
    # print(X[:5])
    # print(X[0][:2000])
    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['category'])
    # print(y[:10])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y = y.reshape((6000, 1))
    y_one_hot = encoder.fit_transform(y)
    # print(y_one_hot[:10])    

    # compute the maximum length of sequence
    # maxlen = 0
    # for sequence in [x[:80000].split(" ") for x in X]:
    #     if len(sequence) > maxlen:
    #         maxlen = len(sequence)
    # print('maximum maxlen: %s' % str(maxlen) )
    acc_sequence_size = []
    acc_val_sequence_size = []
    for sequence_size in [10, 20, 30, 50, 100, 200, 400, 800, 1000]:
        from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=sequence_size)
        text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
        vectorizer.adapt(text_ds)
        # print(vectorizer.get_vocabulary()[:10])
        # print(len(vectorizer.get_vocabulary()))
        x_ix = vectorizer(np.array([[s[:6000]] for s in X])).numpy()
        # print(X[0])
        # print(x_ix[:10])
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(x_ix, y_one_hot, test_size=0.33, random_state=42)
        # print(y_train[:5])
        from sklearn.model_selection import KFold
        acc_10_CV = []
        val_acc_10_CV = []
        for train_index, test_index in KFold(10).split(x_ix):
            X_train, X_test= x_ix[train_index], x_ix[test_index]
            y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
            model = create_model(vectorizer = vectorizer, maxlen = sequence_size, region_size = 3)
            history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
            history_dict = history.history
            acc_10_CV.append(history_dict['accuracy'][-1])
            val_acc_10_CV.append(history_dict['val_accuracy'][-1])
        mean_acc = np.mean(acc_10_CV)
        mean_acc_val = np.mean(val_acc_10_CV)
        print('sequence size: ' + str(sequence_size))
        print(mean_acc)
        print(mean_acc_val)
        acc_sequence_size.append(mean_acc)
        acc_val_sequence_size.append(mean_acc_val)
    print('sequence tune final results: ')
    print(acc_sequence_size)
    print(acc_val_sequence_size)
    df4 = pd.DataFrame(columns = ['acc_sequence_size', 'acc_val_sequence_size'])
    df4['acc_sequence_size'] = acc_sequence_size
    df4['acc_val_sequence_size'] = acc_val_sequence_size
    df4.to_csv('CNN_sequence.csv',  header = True)

def tuning_CNN_sequence_size():
    """ 
    This function works for tuning the CNN model in different sequence size.
    """
    df = pd.read_csv('Queen.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    X = df['text']

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['category'])
    # print(y[:10])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y = y.reshape((6000, 1))
    y_one_hot = encoder.fit_transform(y)
    # print(y_one_hot[:10])    

    # compute the maximum length of sequence
    # maxlen = 0
    # for sequence in [x[:80000].split(" ") for x in X]:
    #     if len(sequence) > maxlen:
    #         maxlen = len(sequence)
    # print('maximum maxlen: %s' % str(maxlen) )
    acc_sequence_size = []
    acc_val_sequence_size = []
    for sequence_size in [10, 20, 30, 50, 100, 200, 400, 800, 1000]:
        from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
        vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=sequence_size)
        text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
        vectorizer.adapt(text_ds)
        # print(vectorizer.get_vocabulary()[:10])
        # print(len(vectorizer.get_vocabulary()))
        x_ix = vectorizer(np.array([[s[:6000]] for s in X])).numpy()
        # print(X[0])
        # print(x_ix[:10])
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(x_ix, y_one_hot, test_size=0.33, random_state=42)
        # print(y_train[:5])
        from sklearn.model_selection import KFold
        acc_10_CV = []
        val_acc_10_CV = []
        for train_index, test_index in KFold(10).split(x_ix):
            X_train, X_test= x_ix[train_index], x_ix[test_index]
            y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
            model = create_model(vectorizer = vectorizer, maxlen = sequence_size, region_size = 3)
            history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
            history_dict = history.history
            acc_10_CV.append(history_dict['accuracy'][-1])
            val_acc_10_CV.append(history_dict['val_accuracy'][-1])
        mean_acc = np.mean(acc_10_CV)
        mean_acc_val = np.mean(val_acc_10_CV)
        print('sequence size: ' + str(sequence_size))
        print(mean_acc)
        print(mean_acc_val)
        acc_sequence_size.append(mean_acc)
        acc_val_sequence_size.append(mean_acc_val)
    print('sequence tune final results: ')
    print(acc_sequence_size)
    print(acc_val_sequence_size)
    df4 = pd.DataFrame(columns = ['acc_sequence_size', 'acc_val_sequence_size'])
    df4['acc_sequence_size'] = acc_sequence_size
    df4['acc_val_sequence_size'] = acc_val_sequence_size
    df4.to_csv('CNN_sequence.csv',  header = True)

def tuning_CNN_region_size():
    """ 
    This function works for tuning the CNN model in region (window) sizes.
    """
    df = pd.read_csv('Queen.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    X = df['text']

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['category'])
    # print(y[:10])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y = y.reshape((6000, 1))
    y_one_hot = encoder.fit_transform(y)
    # print(y_one_hot[:10])    

    # compute the maximum length of sequence
    # maxlen = 0
    # for sequence in [x[:80000].split(" ") for x in X]:
    #     if len(sequence) > maxlen:
    #         maxlen = len(sequence)
    # print('maximum maxlen: %s' % str(maxlen) )


    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1000)
    text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
    vectorizer.adapt(text_ds)
    # print(vectorizer.get_vocabulary()[:10])
    # print(len(vectorizer.get_vocabulary()))
    x_ix = vectorizer(np.array([[s[:6000]] for s in X])).numpy()
    # print(X[0])
    # print(x_ix[:10])
    from sklearn.model_selection import KFold
    acc_region_size = []
    acc_val_region_size = []
    region_sizes = [1, 3, 5, 7, 10, 15, 20, 25, 30]
    for region_size in region_sizes:
        acc_10_CV = []
        val_acc_10_CV = []
        for train_index, test_index in KFold(10).split(x_ix):
            X_train, X_test= x_ix[train_index], x_ix[test_index]
            y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
            model = create_model(vectorizer = vectorizer, maxlen = 1000, region_size = region_size)
            history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
            history_dict = history.history
            acc_10_CV.append(history_dict['accuracy'][-1])
            val_acc_10_CV.append(history_dict['val_accuracy'][-1])
        mean_acc = np.mean(acc_10_CV)
        mean_acc_val = np.mean(val_acc_10_CV)
        print('region size: ' + str(region_size))
        print(mean_acc)
        print(mean_acc_val)
        acc_region_size.append(mean_acc)
        acc_val_region_size.append(mean_acc_val)
    print('region tune final results: ')
    print(acc_region_size)
    print(acc_val_region_size)
    df2 = pd.DataFrame(columns = ['acc_region_size', 'acc_val_region_size'])
    df2['acc_region_size'] = acc_region_size
    df2['acc_val_region_size'] = acc_val_region_size
    df2.to_csv('CNN_region.csv',  header = True)

    from sklearn.model_selection import KFold
    acc_featuremaps_size = []
    acc_val_featuremaps_size = []
    featuremaps_sizes = [10, 20, 30, 50, 100, 200, 400, 800, 1000]
    for featuremaps_size in featuremaps_sizes:
        acc_10_CV = []
        val_acc_10_CV = []
        for train_index, test_index in KFold(10).split(x_ix):
            X_train, X_test= x_ix[train_index], x_ix[test_index]
            y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
            model = create_model(vectorizer = vectorizer, maxlen = 1000, region_size = 3, n_feature_maps=featuremaps_size)
            history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
            history_dict = history.history
            acc_10_CV.append(history_dict['accuracy'][-1])
            val_acc_10_CV.append(history_dict['val_accuracy'][-1])
        mean_acc = np.mean(acc_10_CV)
        mean_acc_val = np.mean(val_acc_10_CV)
        print('featuremaps size: ' + str(featuremaps_size))
        print(mean_acc)
        print(mean_acc_val)
        acc_featuremaps_size.append(mean_acc)
        acc_val_featuremaps_size.append(mean_acc_val)
    print('featuremaps tune final results: ')
    print(acc_featuremaps_size)
    print(acc_val_featuremaps_size)
    df3 = pd.DataFrame(columns = ['acc_featuremaps_size', 'acc_val_featuremaps_size'])
    df3['acc_featuremaps_size'] = acc_featuremaps_size
    df3['acc_val_featuremaps_size'] = acc_val_featuremaps_size
    df3.to_csv('CNN_featuremaps.csv',  header = True)

def tuning_CNN_featuremaps_size():
    """ 
    This function works for tuning the CNN model in featuremaps sizes.
    """
    df = pd.read_csv('Queen.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    X = df['text']

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['category'])
    # print(y[:10])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y = y.reshape((6000, 1))
    y_one_hot = encoder.fit_transform(y)
    # print(y_one_hot[:10])    

    # compute the maximum length of sequence
    # maxlen = 0
    # for sequence in [x[:80000].split(" ") for x in X]:
    #     if len(sequence) > maxlen:
    #         maxlen = len(sequence)
    # print('maximum maxlen: %s' % str(maxlen) )


    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=1000)
    text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
    vectorizer.adapt(text_ds)
    # print(vectorizer.get_vocabulary()[:10])
    # print(len(vectorizer.get_vocabulary()))
    x_ix = vectorizer(np.array([[s[:6000]] for s in X])).numpy()
    # print(X[0])
    # print(x_ix[:10])

    from sklearn.model_selection import KFold
    acc_featuremaps_size = []
    acc_val_featuremaps_size = []
    featuremaps_sizes = [10, 20, 30, 50, 100, 200, 400, 800, 1000]
    for featuremaps_size in featuremaps_sizes:
        acc_10_CV = []
        val_acc_10_CV = []
        for train_index, test_index in KFold(10).split(x_ix):
            X_train, X_test= x_ix[train_index], x_ix[test_index]
            y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
            model = create_model(vectorizer = vectorizer, maxlen = 1000, region_size = 3, n_feature_maps=featuremaps_size)
            history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
            history_dict = history.history
            acc_10_CV.append(history_dict['accuracy'][-1])
            val_acc_10_CV.append(history_dict['val_accuracy'][-1])
        mean_acc = np.mean(acc_10_CV)
        mean_acc_val = np.mean(val_acc_10_CV)
        print('featuremaps size: ' + str(featuremaps_size))
        print(mean_acc)
        print(mean_acc_val)
        acc_featuremaps_size.append(mean_acc)
        acc_val_featuremaps_size.append(mean_acc_val)
    print('featuremaps tune final results: ')
    print(acc_featuremaps_size)
    print(acc_val_featuremaps_size)
    df3 = pd.DataFrame(columns = ['acc_featuremaps_size', 'acc_val_featuremaps_size'])
    df3['acc_featuremaps_size'] = acc_featuremaps_size
    df3['acc_val_featuremaps_size'] = acc_val_featuremaps_size
    df3.to_csv('CNN_featuremaps.csv',  header = True)