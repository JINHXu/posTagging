#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    <Please insert your name and the honor code here.>
"""
import random
from _csv import reader

from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import preprocessing
from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, confusion_matrix
import numpy as np
import nltk
import pandas as pd
import tf as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical


def read_data(treebank, shuffle=True, lowercase=True,
        tags=None):
    """ Read a CoNLL-U formatted treebank, return words and POS tags.

    Parameters:
    -----------
    treebank:  The path to the treebank (individual file).
    shuffle:   If True, shuffle the word-POS pairs before returning
    lowercase: Convert words (tokens) to lowercase
    tags:      If not 'None', return only the pairs where POS tags in tags

    Returns: (a tuple)
    -----------
    words:      A list of words.
    pos:        Corresponding POS tags.
    """

    word_pos = []

    with open(treebank, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # if lowercase true, set everything to lowercase
        if lowercase:
            lines = [x.lower() for x in lines]
        # if shuffle is true, shuffle lines
        if shuffle:
            random.shuffle(lines)
        # for all lines that are not empty or start with #, split lines into tokens, and get all contents
        # of second column (second word) and fourth column (fourth word)
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            content = line.split()
            words = content[1]
            pos_tag = content[3]

            # check if specific word is already used, by using an empty list, only appendng words and their pos-tags
            # as tuples if they are not yet in the list
            if (words, pos_tag) in word_pos:
                continue
            else:
                if tags is not None:
                    if pos_tag in tags:
                        word_pos.append((words, pos_tag))
                    else:
                        continue
                else:
                    word_pos.append((words, pos_tag))

    #print(word_pos)
    # receive all words, and all tags in list-variable for words and tags separately
    words = [word[0] for word in word_pos]
    pos_tags = [tags[1] for tags in word_pos]

    return words, pos_tags

class WordEncoder:
    """An encoder for a sequence of words.

    The encoder encodes each word as a sequence of one-hot characters.
    The words that are longer than 'maxlen' is truncated, and
    the words that are shorter are padded with 0-vectors.
    Two special symbols, <s> and </s> (beginning of sequence and
    end of sequence) should be prepended and appended to every word
    before padding or truncation. You should also reserve a symbol for
    unknown characters (distinct from the padding).

    The result is either a 2D vector, where all character vectors
    (including padding) of a word are concatenated as a single vector,
    o a 3D vector with a separate vector for each character (see
    the description of 'transform' below and the assignment sheet
    for more detail.

    Parameters:
    -----------
    maxlen:  The length that each word (including <s> and </s>) is padded
             or truncated. If not specified, the fit() method should
             set it to cover the longest word in the training set.
    """

    def __init__(self, maxlen = None):
        ### part of 5.2
        self._maxlen = maxlen
        self._singleChars = {}
        self._total_number_chars = len(self._singleChars)


    def add_special_tokens(self, words):
        word_list = []
        for word in words:
            word_list.append("<s>" + word + "</s>")
        return word_list

    def fit(self, words):
        """Fit the encoder using words.

        All collection of information/statistics from the training set
        should be done here.

        Parameters:
        -----------
        words:  The input words used for training.

        Returns: None
        """

        ### part of 5.2

        sortedwords = sorted(words, key=len)
        long_word = sortedwords[-1]
        longest_word = len(sortedwords[-1]) + 2  # longest word +2 since <s> and </s> count as one character respectively

        # the fit() method should set maxLen, to cover the longest word in the training set
        if self._maxlen is None:
            self._maxlen = longest_word

        # separate number of each character
        self._singleChars["<s>"] = 0  # special character <s> = 0
        self._singleChars["</s>"] = 1  # special character </s> = 1
        self._singleChars["UNK"] = 2  # special character for any unknown character in test data = 2
        index = 3
        char_list = []
        # for any word in word_list
        for word in words:
            # for any char in each word
            for char in word:
                # if char is new and is not in char_list yet, append to char_list, and add char to dictionary
                # with index as value
                if char not in char_list and self._singleChars:
                    char_list.append(char)
                    self._singleChars[char] = index
                    index += 1  # increment index by 1 so that each char receives individual index
                else:
                    continue

        self._total_number_chars = len(self._singleChars)


    def transform(self, words, pad='right', flat=True):
        """ Transform a sequence of words to a sequence of one-hot vectors.

        Transform each character in each word to its one-hot representation,
        combine them into a larger sequence, and return.

        The returned sequences formatted as a numpy array with shape
        (n_words, max_wordlen * n_chars) if argument 'flat' is true,
        (n_words, max_wordlen, n_chars) otherwise. In both cases
        n_words is the number of words, max_wordlen is the maximum
        word length either set in the constructor, or determined in
        fit(), and n_chars is the number of unique characters.

        Parameters:
        -----------
        words:  The input words to encode
        pad:    Padding direction, either 'right' or 'left'
        flat:   Whether to return a 3D array or a 2D array (see above
                for explanation)

        Returns: (a tuple)
        -----------
        encoded_data:  encoding the input words (a 2D or 3D numpy array)
        """
        ### part of 5.2

        # append special characters to words
        word_list = self.add_special_tokens(words)

        word_to_array = np.zeros((len(words), self._maxlen, self._total_number_chars))

        for i, word in enumerate(word_list):
            # truncate if word longer than maxlen, but from the beginning,
            # since the end can give information what word type it is, e.g. ly - adverb...
            if len(word) > self._maxlen:
                word = word[-self._maxlen:]

            if len(word) < self._maxlen and pad == "left":
                # number to pad
                number_to_pad = self._maxlen - len(word)
                number_to_pad * [-1] + word  # -1 is symbol to pad

            for j, character in enumerate(word):
                if character == -1:
                    continue
                elif j==0 and (word[0:3] == "<s>"):
                    character = "<s>"
                    index = self._singleChars[character]
                    word_to_array[i, j, index] = 1
                    continue
                elif j==1 or j==2 and (word[1:3] == "s>"):
                    continue
                elif j==len(word)-4 and (word[len(word)-4:] == "</s>"):
                    character = "</s>"
                    index = self._singleChars[character]
                    word_to_array[i, j, index] = 1
                    continue
                elif j==(len(word)-3) or j==(len(word)-2) or j==(len(word)-1) and (word[(len(word)-3):len(word)] == "/s>"):
                    continue
                elif character in self._singleChars:
                    index = self._singleChars[character]
                    word_to_array[i, j, index] = 1

                else:
                    character = "UNK"
                    index = self._singleChars[character]
                    word_to_array[i, j, index] = 1

        if flat:
            word_to_array = np.reshape(word_to_array, (len(words), self._total_number_chars*self._maxlen))

        return word_to_array

def precision_recall_f1_confusionMatrix(y_test, y_pred):
    print(f'macro precision score: \n {precision_score(y_test, y_pred, average="macro")}')
    print(f'macro recall score: \n {recall_score(y_test, y_pred, average="macro")}')
    print(f'macro f1-score: \n {f1_score(y_test, y_pred, average="macro")}')
    print(f'confusion matrix: \n {confusion_matrix(y_test, y_pred)}')

def training_phase_both_models(train_x, train_pos, test_x, test_pos, filepath, model):
    # define callbacks to early stop training and save best model
    callbacks = [EarlyStopping(monitor='val_loss', patience=4),
                 ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True)]
    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # train model, epochs (iterations = 50), batch-size=250 (number of observations per batch),
    # validation split 20% (uses 20% of training samples as optimization)
    fitted_model = model.fit(train_x, train_pos, epochs=50, callbacks=callbacks, validation_split=0.2)

    # # test model after training
    # test_results = model.evaluate(test_x, test_pos, verbose=1)
    # print(test_results)

    return fitted_model

def retrain_models(train_x, train_pos, test_x, test_pos, model, best_epochs):

    # re-train model on entire data-set
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_pos, epochs=best_epochs)
    y_pred = model.predict(test_x)
    return y_pred


def encode_train_pos(train_pos):
    # encode train_pos
    labeler = preprocessing.LabelBinarizer()
    labeler.fit(train_pos)
    return labeler

def train_test_mlp(train_x, train_pos, test_x, test_pos):
    """Train and test MLP model predicting POS tags from given encoded words.

    Parameters:
    -----------
    train_x:    A sequence of words encoded as described above
                (a 2D numpy array)
    train_pos:  The list of list of POS tags corresponding to each row
                of train_x.
    test_x, test_pos: As train_x, train_pos, for the test set.

    Returns: None
    """
    ### 5.3 - you may want to implement parts of the solution
    ###       in other functions so that you share the common
    ###       code between 5.3 and 5.4



    # convert target classes to categorical ones
    # encode train_pos
    labeler = encode_train_pos(train_pos)
    train_pos = labeler.transform(train_pos)
    print(type(train_pos))

    # input shape
    input_shape = (train_x.shape[1],)

    # output shape
    output_shape = train_pos.shape[1]

    # define Keras model
    model = Sequential()
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    # add output layer
    model.add(Dense(units=output_shape, activation="softmax"))


    # file to save weights of callback for model before retraining
    before_filepath = "best_mlp_before_retraining.hdf5"
    fitted_model = training_phase_both_models(train_x, train_pos, test_x, test_pos, before_filepath, model)
    print(type(fitted_model))

    # best epoch
    loss_hist = fitted_model.history['val_loss']
    best_epoch = np.argmin(loss_hist)

    # file to save weights of callback for model after retraining
    y_pred = retrain_models(train_x, train_pos, test_x, test_pos, model, best_epoch)
    y_pred = labeler.inverse_transform(y_pred)

    # print macro averaged precision, recall, f1-score and confusion matrix on test-set
    precision_recall_f1_confusionMatrix(test_pos, y_pred)


def train_test_rnn(trn_x, trn_pos, tst_x, tst_pos):
    """
    Train and test RNN model predicting POS tags from given encoded words.

    Parameters:
    -----------
    train_x:    A sequence of words encoded as described above
                (a 3D numpy array)
    train_pos:  The list of list of POS tags corresponding to each row
                of train_x.
    test_x, test_pos: As train_x, train_pos, for the test set.

    Returns: None
    """
    ### 5.4
    # encode trn_pos
    labeler = encode_train_pos(trn_pos)
    trn_pos = labeler.transform(trn_pos)

    # input shape
    input_shape = (trn_x.shape[0],)

    # output shape
    output_shape = trn_pos.shape[1]



    model = Sequential()
    model.add(Embedding(input_dim=trn_x.shape[0], output_dim=output_shape))
    model.add(LSTM(units=64, activation="relu"))
    model.add(Dense(units=output_shape, activation="softmax"))

    before_filepath = "best_lstm_before_retraining.hdf5"
    fitted_model = training_phase_both_models(trn_x, trn_pos, tst_x, tst_pos, before_filepath, model)

    # best epoch
    loss_hist = fitted_model.history['val_loss']
    best_epoch = np.argmin(loss_hist)

    # file to save weights of callback for model after retraining

    y_pred = retrain_models(trn_x, trn_pos, tst_x, tst_pos, model, best_epoch)
    y_pred = labeler.inverse_transform(y_pred)

    # print macro averaged precision, recall, f1-score and confusion matrix on test-set
    precision_recall_f1_confusionMatrix(test_pos, y_pred)


if __name__ == '__main__':
    ### Not checked for grading,
    ### but remember that before calling train_test_mlp() and 
    ### train_test_rnn(), you need to split the as training and test
    ### set properly.

    #
    # dataset downloaded from https://github.com/UniversalDependencies/UD_English-ParTUT
    words, pos_tags = read_data("en_partut-ud-train.conllu")
    test_words, test_pos = read_data("en_partut-ud-test.conllu")
    encoder = WordEncoder()
    encoder.fit(words)
    encoded_array = encoder.transform(words)
    test_encoded_array = encoder.transform(test_words)

    # X_train, X_test, y_train, y_test = train_test_split(encoded_array, pos_tags, test_size=0.2, random_state=1)

    # train_test_mlp(encoded_array, pos_tags, test_encoded_array, test_pos)
    train_test_rnn(encoded_array, pos_tags, test_encoded_array, test_pos)


