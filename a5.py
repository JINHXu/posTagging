#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    Course:      Statistical Language processing - SS2020
    Assignment:  A5
    Author(s):   Anna-Sophie Bartle, Jinghua Xu
    Description: experiment with NN
    
    Honor Code:  We pledge that this program represents our own work.
"""

import random
import numpy as np

from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


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
    # list of word_pos pairs
    word_pos = []

    with open(treebank, 'r') as f:
        lines = f.readlines()
        if shuffle:
            random.shuffle(lines)
        for line in lines:
            # skip blank lines and comments
            if not line.strip() or line.startswith('#'):
                continue
            vs = line.split('\t')
            # skip words (and their POS tags) that are part of a multi-word token and empty nodes
            idx = vs[0]
            if '-' in idx or '.' in idx:
                continue
            word = vs[1]
            pos = vs[3]
            if lowercase:
                word = word.lower()
            # unique pairs
            if (word, pos) in word_pos:
                continue
            else:
                if tags is not None:
                    if pos in tags:
                        word_pos.append((word, pos))
                    else:
                        continue
                else:
                    word_pos.append((word, pos))
    words = [x[0] for x in word_pos]
    pos = [x[1] for x in word_pos]
    return words, pos


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

    def __init__(self, maxlen=None):
        # to be set up in fit()
        self._maxlen = maxlen
        self._char2idx = dict()
        self._nchars = len(self._char2idx)

    def fit(self, words):
        """Fit the encoder using words.

        All collection of information/statistics from the training set
        should be done here.

        Parameters:
        -----------
        words:  The input words used for training.

        Returns: None
        """
        setUPmaxlen = False
        if self._maxlen is None:
            self._maxlen = 0
            setUPmaxlen = True

        # special symbols
        self._char2idx['<s>'] = 0
        self._char2idx['</s>'] = 1
        # reserve for unknown chararacters
        self._char2idx['uk'] = 2

        # current index
        idx = 3
        # chars in words
        for word in words:
            if len(word) > self._maxlen and setUPmaxlen:
                self._maxlen = len(word)
            for char in word:
                if char not in self._char2idx:
                    self._char2idx[char] = idx
                    idx += 1
        self._nchars = len(self._char2idx)

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

        # params check
        if isinstance(flat, bool) and (pad == 'right' or pad == 'left'):
            pass
        else:
            raise ValueError(
                "Illegal Argument! pad can only be 'right' or 'left', flat has to be bool!")
        encoded_words = []
        for word in words:
            word = list(word)
            encoded_word = []
            # prepend special char
            word.insert(0, '<s>')
            # append special char
            word.append('</s>')
            if len(word) > self._maxlen:
                # truncation
                word = word[:self._maxlen]
            for char in word:
                char_vec = [0]*self._nchars
                if char in self._char2idx:
                    idx = self._char2idx[char]
                    char_vec[idx] = 1
                else:
                    # unknown char
                    char = 'uk'
                    idx = self._char2idx[char]
                    char_vec[idx] = 1
                if flat:
                    encoded_word = encoded_word + char_vec
                else:
                    encoded_word.append(char_vec)
            if len(word) < self._maxlen:
                # padding
                padding = [0]*self._nchars
                if pad == 'right':
                    for _ in range(self._maxlen-len(word)):
                        if flat:
                            encoded_word = encoded_word + padding
                        else:
                            encoded_word.append(padding)
                else:
                    for _ in range(self._maxlen-len(word)):
                        if flat:
                            encoded_word = padding + encoded_word
                        else:
                            encoded_word.insert(0, padding)
            encoded_words.append(encoded_word)
        return np.array(encoded_words)


def print_stats(test_pos, y_test_pred):
    """Print out macro-averaged precision, recall, F1 scores, and the confusion matrix on the test set
    Parameters:
    -----------
        test_pos:   pos tags in test data
        y_test_pred:    predicted y(pos) of test data

    Returns: None
    """
    print(
        f'macro-averaged precision: {precision_score(test_pos, y_test_pred, average="macro")}')
    print(
        f'macro-averaged recall: {recall_score(test_pos, y_test_pred, average="macro")}')
    print(
        f'macro-averaged f-1: {f1_score(test_pos, y_test_pred, average="macro")}')
    print(f'confusion-matrix:\n {confusion_matrix(test_pos, y_test_pred)}')


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
    # encode train_pos
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_pos)
    encoded_train_pos = lb.transform(train_pos)

    # output shape
    output_layer_units = encoded_train_pos.shape[1]

    # input shape
    input_shape = (train_x.shape[1],)

    mlp = Sequential()
    # hidden layer
    mlp.add(Dense(units=64, activation='relu', input_shape=input_shape))
    # output layer
    mlp.add(Dense(units=output_layer_units, activation='softmax'))

    mlp.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    hist = mlp.fit(train_x, encoded_train_pos, epochs=50, validation_split=0.2)

    # the best epoch
    losses = hist.history['loss']
    best_epoch = losses.index(min(losses))

    # re-train the model (from scratch) using the full training set up to the best epoch determined earlier
    mlp.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.fit(train_x, encoded_train_pos, epochs=best_epoch)

    # print out macro-averaged precision, recall, F1 scores, and the confusion matrix on the test set
    y_test_pred = lb.inverse_transform(mlp.predict(test_x))

    # print stats
    print_stats(test_pos, y_test_pred)


def train_test_rnn(trn_x, trn_pos, tst_x, tst_pos):
    """Train and test RNN model predicting POS tags from given encoded words.
    Parameters:
    -----------
    train_x:    A sequence of words encoded as described above
                (a 3D numpy array)
    train_pos:  The list of list of POS tags corresponding to each row
                of train_x.
    test_x, test_pos: As train_x, train_pos, for the test set.
    Returns: None
    """
    # encode train_pos
    lb = preprocessing.LabelBinarizer()
    lb.fit(trn_pos)
    encoded_train_pos = lb.transform(trn_pos)

    # output shape
    output_dim = encoded_train_pos.shape[1]

    rnn = Sequential()
    rnn.add(LSTM(64, input_shape=(
        trn_x.shape[1], trn_x.shape[2]), activation='relu'))
    rnn.add(Dense(output_dim, activation='softmax'))

    rnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    hist = rnn.fit(trn_x, encoded_train_pos, epochs=50, validation_split=0.2)

    # the best epoch
    losses = hist.history['loss']
    best_epoch = losses.index(min(losses))

    # re-train the model (from scratch) using the full training set up to the best epoch determined earlier
    rnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    rnn.fit(trn_x, encoded_train_pos, epochs=best_epoch)

    # print out macro-averaged precision, recall, F1 scores, and the confusion matrix on the test set
    y_test_pred = lb.inverse_transform(rnn.predict(tst_x))

    # print stats
    print_stats(test_pos, y_test_pred)


if __name__ == '__main__':
    #
    # dataset downloaded from https://github.com/UniversalDependencies/UD_English-ParTUT
    words, pos_tags = read_data("en_partut-ud-train.conllu")
    test_words, test_pos = read_data("en_partut-ud-test.conllu")
    encoder = WordEncoder()
    encoder.fit(words)

    # 5.3
    encoded_array = encoder.transform(words)
    test_encoded_array = encoder.transform(test_words)
    train_test_mlp(encoded_array, pos_tags, test_encoded_array, test_pos)

    print("-------------------------------------------------------------------------------------------------------------")

    # 5.4
    rnn_train_words = encoder.transform(words, flat=False)
    rnn_test_words = encoder.transform(test_words, flat=False)
    train_test_rnn(rnn_train_words, pos_tags, rnn_test_words, test_pos)
