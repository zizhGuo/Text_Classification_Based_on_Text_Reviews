import tensorflow as tf
import tensorflow.keras as keras
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import numpy as np
import pandas as pd
# import stem_lemmatize.py


def create_model(vectorizer, maxlen, region_size, trainable = False, n_feature_maps = 128):
    """ 
    This function uses TensorFlow Keras Sequential model to create a basic Yoon Kim CNN model for text classification.
        Params:
            @vectorizer: a TextVectorization class imported from Tensorflow keras preprocessing layer
                        It works to tokenize the textual samples.
            @maxlen: the sequence(one sample review) length (characters)
            @region_size: the window(convolution kernel) size
            @trainable: static or non-static that decides whether the word embeddings participate the training process or not
            @n_feature_maps: the number of windows/kernels (patterns)
        Return:
            the configured TensorFlow Keras model 
    """
    # 用gensim载入预处理glove模型 using gensim loaded GloVe word vectors
    # 词-向量 字典 WORD-EMBEDDING paired dictionary
    from gensim.models import KeyedVectors
    # glove_model = KeyedVectors.load('pretrained_glove.bin')
    glove_model = KeyedVectors.load_word2vec_format('sg_wv.bin', binary=True)

    # 用gensim载入预处理google w2v模型
    # start = datetime.datetime.now()
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # print(word2vec_model.wv['food'])
    # end = datetime.datetime.now()
    # print('Load Pretarined Model time spent: %s' % (end-start))

    # initialize the word-index dictionary from vectorizer
    # 词-序号 字典 WORD-INDEX paired dictionary
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    # 序号-向量 字典 INDEX-EMBEDDING paired dictionary
    num_tokens = len(voc) + 2
    embedding_dim = 300
    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        # embedding_vector = word2vec_model.wv[word]
        if word in glove_model:
            embedding_vector = glove_model[word]
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            embedding_vector = [0.000] * embedding_dim
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    from tensorflow.keras.layers import Embedding
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=trainable,
        input_length= maxlen
    )

    from tensorflow.keras import layers

    # input layer placeholder
    int_sequences_input = keras.Input(shape=(None,), dtype="float32")
    print(int_sequences_input.shape)

    # embedding layer INDEX-EMBEDDING dictionary
    embedded_sequences = embedding_layer(int_sequences_input)
    print(embedded_sequences.shape)

    # Convolution layer
    x1 = layers.Conv1D(n_feature_maps, region_size, activation="relu")(embedded_sequences)
    print(x1.shape)

    # GlobalMax pooling layer
    x1 = layers.GlobalMaxPooling1D()(x1)
    print(x1.shape)
    
    # x2 = layers.Conv1D(128, 4, activation="relu")(embedded_sequences)
    # print(x2.shape)
    # x2 = layers.GlobalMaxPooling1D()(x2)
    # print(x2.shape)

    # x3 = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
    # print(x3.shape)
    # x3 = layers.GlobalMaxPooling1D()(x3)
    # print(x3.shape)

    # y = layers.concatenate([x1, x2, x3], axis=-1)
    # print(y.shape)

    # Flatten layer
    y = layers.Flatten()(x1)
    print(y.shape)

    # Dropout layer
    y = layers.Dropout(0.5)(y)
    print(y.shape)

    # SoftMax layer
    output = layers.Dense(3, activation='softmax')(y)
    model = keras.Model(int_sequences_input, output)
    print(model.summary())

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model