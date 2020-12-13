import tensorflow as tf
import tensorflow.keras as keras
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import numpy as np
import pandas as pd
# import stem_lemmatize.py


def create_RNN_model(vectorizer, maxlen):
    """ 
    This function uses TensorFlow Keras Sequential model to create a basic RNN Encoder model using memory unit of GRU or Bidirectional LSTM
        Params:
            @vectorizer: a TextVectorization class imported from Tensorflow keras preprocessing layer
                        It works to tokenize the textual samples.
            @maxlen: the sequence(one sample review) length (characters)
        Return:
            the configured TensorFlow Keras model 
    """
    # 用gensim载入预处理glove模型
    # 词-向量 字典
    from gensim.models import KeyedVectors
    glove_model = KeyedVectors.load('pretrained_glove.bin')
    # glove_model = KeyedVectors.load_word2vec_format('test_glove.txt')
    # glove_model = KeyedVectors.load_word2vec_format("sg_wv.bin")

    # initialize the word-index dictionary from vectorizer
    # 词-序号 字典
    voc = vectorizer.get_vocabulary()
    print(voc[:10])
    word_index = dict(zip(voc, range(len(voc))))

    # 序号-向量 字典
    num_tokens = len(voc) + 2
    embedding_dim = 300
    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        # embedding_vector = word2vec_model.wv[word]
        if word in glove_model:
            embedding_vector = glove_model.wv[word]
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            # embedding_vector = [0.000] * embedding_dim
            embedding_matrix[i] = glove_model.wv['unk']
            # embedding_matrix[i] = glove_model.wv['[UNK]']
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    from tensorflow.keras.layers import Embedding
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        # embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        embeddings_initializer="uniform",
        # weights = [embedding_matrix],
        trainable = True,
        input_length= maxlen
    )

    from tensorflow.keras import layers
    # int_sequences_input = keras.Input(shape=(None,), dtype="float32")
    int_sequences_input = keras.Input(shape=(maxlen,), dtype="float32")
    print(int_sequences_input.shape)
    embedded_sequences = embedding_layer(int_sequences_input)
    print(embedded_sequences.shape)

    # state_h = layers.Bidirectional(layers.LSTM(64))(embedded_sequences)
    # state_h = layers.LSTM(64)(embedded_sequences)
    gru1, state_h = layers.GRU(64, return_sequences=True, return_state=True)(embedded_sequences)
    # gru1, state_h = layers.GRU(64, return_sequences=True, return_state=True)(gru1)
    print(state_h.shape)

    output = layers.Dense(3, activation='softmax')(state_h)

    model = keras.Model(int_sequences_input, output)
    print(model.summary())

    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model_rnn.png',show_shapes=True,show_layer_names=True)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model