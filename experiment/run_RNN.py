import tensorflow as tf
import tensorflow.keras as keras
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# import create_RNN_model

def main():
    df = pd.read_csv('IS_new.csv')
    df = df.drop(columns = ['Unnamed: 0'])
    X = df['text']
    # print(X[:10])

    from sklearn.preprocessing import LabelEncoder
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['category'])
    # print(y[:3])
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y = y.reshape((6000, 1))
    y_one_hot = encoder.fit_transform(y)
    # print(y_one_hot[:3])

    from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
    vectorizer = TextVectorization(max_tokens=200000, output_sequence_length=1000)
    text_ds = tf.data.Dataset.from_tensor_slices(X).batch(64) # transform to Tensors Dataset
    vectorizer.adapt(text_ds) # Use dataset to update the vocabularies and sets word indices
    print(vectorizer.get_vocabulary()[:10])
    print(len(vectorizer.get_vocabulary()))
    x_ix = vectorizer(np.array([[s[:6000]] for s in X])).numpy() # input X and convert strings into list of indices (better use Input Layer)
    # x_ix = vectorizer(X).numpy() # input X and convert strings into list of indices (better use Input Layer)
    # print(X[0])
    print(x_ix[:10]) # unkown words are marked as index: 1 which represents <UNK>
    
    # test if model can be made at such big size vocabulary or not
    # model = create_RNN_model(vectorizer = vectorizer, maxlen = 1000)
    
    from sklearn.model_selection import KFold
    acc_10_CV = []
    val_acc_10_CV = []
    for train_index, test_index in KFold(10).split(x_ix):
        X_train, X_test= x_ix[train_index], x_ix[test_index]
        y_train, y_test= y_one_hot[train_index], y_one_hot[test_index]            
        model = create_RNN_model(vectorizer = vectorizer, maxlen = 1000)
        history = model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_test, y_test))
        history_dict = history.history
        acc_10_CV.append(history_dict['accuracy'][-1])
        val_acc_10_CV.append(history_dict['val_accuracy'][-1])
    mean_acc = np.mean(acc_10_CV)
    mean_acc_val = np.mean(val_acc_10_CV)
    print(mean_acc)
    print(mean_acc_val)

if __name__ == "__main__":
    main()