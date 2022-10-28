from numpy import unique
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers.merging import concatenate
from keras.utils import plot_model
from fastai.basics import *
from fastai.tabular.all import *
import warnings
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler # This is also a normalizer
from keras.preprocessing.sequence import TimeseriesGenerator # Puts the data in time series format
from tensorflow import keras
warnings.filterwarnings("ignore")


# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename, header=None)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, :-1]
    y = dataset[:, -1]
    # format all fields as string
    X = X.astype(str)
    # reshape target to be a 2d array
    y = y.reshape((len(y), 1))
    return X, y


# prepare input data
def prepare_inputs(X_train, X_test):
    X_train_enc, X_test_enc = list(), list()
    # label encode each column
    for i in range(X_train.shape[1]):
        le = LabelEncoder()
        le.fit(X_train[:, i])
        # encode
        train_enc = le.transform(X_train[:, i])
        test_enc = le.transform(X_test[:, i])
        # store
        X_train_enc.append(train_enc)
        X_test_enc.append(test_enc)
    return X_train_enc, X_test_enc


# This should normalize the input array to a range between 0 and 1
def normalize(arr, t_min=0, t_max=1):
    print("Normalising")
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


# This method seemed to be less efficient and less accurate than the one implemented with KERAS
def encode_data_fastai(df):
    X = df[['companyname']]
    #X = tf.convert_to_tensor(X)
    Y  = df[['amount']]
    test_size = 1000
    df_main, df_test = df.iloc[:-test_size].copy(), df.iloc[-test_size:].copy()
    procs = [Categorify, FillMissing, Normalize]
    cat_names = list(X.columns)

    splits = RandomSplitter(valid_pct=0.3)(range_of(df_main))
    to = TabularPandas(df_main, procs, cat_names, y_names=list(Y.columns), splits=splits)
    dls = to.dataloaders(bs=150)
    dls.show_batch()
    learn = tabular_learner(dls, layers=[500, 500, 500, 250], metrics=accuracy)
    #learn.lr_find()
    learn.fit_one_cycle(40, 1e-1)

    test_dl = learn.dls.test_dl(df_test)
    print(learn.validate(dl=test_dl))

    preds = learn.get_preds(dl=test_dl)
    y_pred = np.argmax(preds[0], axis=1).numpy()
    y_true = [int(i) for i in df_test['amount']]
    print(sum(y_pred == y_true) / len(y_pred))


def encode_data_keras1(X, y):
    # Normalize the output data so it can be better predicted, they need to be translated back to the original format to
    # make sense to the reader though
    X = X.values
    y = y.values
    y = np.array(normalize(y))
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # prepare input data
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    # prepare output data
    # make output 3d
    y_train_enc = y_train.reshape((len(y_train), 1, 1))
    y_test_enc = y_test.reshape((len(y_test), 1, 1))
    # prepare each input head
    in_layers = list()
    em_layers = list()
    for i in range(len(X_train_enc)):
        # calculate the number of unique inputs
        n_labels = len(unique(X_train_enc[i]))
        # define input layer
        in_layer = Input(shape=(1,))
        # define embedding layer
        em_layer = Embedding(n_labels, 10000)(in_layer)
        # store layers
        in_layers.append(in_layer)
        em_layers.append(em_layer)
    # concat all embeddings
    merge = concatenate(em_layers)
    dense1 = Dense(10000, activation='relu', kernel_initializer='random_normal')(merge)
    dense2 = Dense(5000, activation='relu', kernel_initializer='random_normal')(dense1)
    dense3 = Dense(1200, activation='relu', kernel_initializer='random_normal')(dense2)
    dense4 = Dense(600, activation='relu', kernel_initializer='random_normal')(dense3)
    dense5 = Dense(100, activation='relu', kernel_initializer='random_normal')(dense4)
    output = Dense(1, activation='sigmoid')(dense5)
    model = Model(inputs=in_layers, outputs=output)
    # compile the keras model
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    # plot graph
    plot_model(model, show_shapes=True, to_file='embeddings.png')
    # fit the keras model on the dataset
    model.fit(X_train_enc, y_train_enc, epochs=150, batch_size=150, verbose=1)
    # evaluate the keras model
    _, accuracy = model.evaluate(X_test_enc, y_test_enc, verbose=0)
    print('Accuracy: %.2f' % (accuracy * 100))


def encode_data_keras2(X, y):
    ohe_data = pd.get_dummies(X)
    X = X.to_numpy()
    num_of_categories = len(np.unique(X))
    #y = np.array(normalize(y.values))
    X_train, X_test, y_train, y_test = train_test_split(ohe_data, y.values, test_size=0.2)
    embedding_size = int(min(np.ceil(num_of_categories/2), 50))
    print(embedding_size)
    model = tf.keras.Sequential()
    #in_cat_data = keras.layers.Input(shape=(num_of_categories,))
    model.add(keras.layers.Input(shape=(num_of_categories,)))
    #in_num_data = keras.layers.Input(shape=(num_data.shape[1],))
    model.add(keras.layers.Embedding(input_dim=num_of_categories, output_dim=embedding_size))
    #emb = keras.layers.Embedding(input_dim=num_of_categories, output_dim=embedding_size)(in_cat_data)
    #flat = keras.layers.Flatten()(emb)
    n_layers = 3
    size_layer = 50
    for _ in range(n_layers):
        model.add(tf.keras.layers.LSTM(size_layer, activation='relu', return_sequences=True))
    # dense1 = keras.layers.Dense(500, activation=tf.nn.relu)(flat)
    # dense2 = keras.layers.Dense(100, activation=tf.nn.relu)(dense1)
    # dense3 = keras.layers.Dense(50, activation=tf.nn.relu)(dense2)

    model.add(keras.layers.Dense(1, activation=tf.nn.relu))
    # out = keras.layers.Dense(1, activation=tf.nn.relu)(dense3)

    # model = keras.Model(inputs=in_cat_data, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=keras.losses.mean_squared_error,
                  metrics='accuracy')
    print(model.summary())
    model.fit(X_train, y_train, epochs=78, batch_size=100, verbose=1)

    _, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print('Accuracy: %.2f' % (accuracy * 100))