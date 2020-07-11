#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    <Please insert your name and the honor code here.>
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


def precision_recall_f1_confusionMatrix(y_test, y_pred):
    print(f'macro precision score: \n {precision_score(y_test, y_pred, average="macro")}')
    print(f'macro recall score: \n {recall_score(y_test, y_pred, average="macro")}')
    print(f'macro f1-score: \n {f1_score(y_test, y_pred, average="macro")}')
    print(f'confusion matrix: \n {confusion_matrix(y_test, y_pred)}')

def training_phase_both_models(train_x, train_pos, test_x, test_pos, model, filepath=""):
    # define callbacks to early stop training and save best model

    if not filepath:
        callbacks = None
    else:
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0.005, patience=4),
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
    fitted_model = training_phase_both_models(train_x, train_pos, test_x, test_pos, model)

    # best epoch
    loss_hist = fitted_model.history['loss']
    best_epoch = np.argmin(loss_hist)
    print(best_epoch)

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
    input_shape = (trn_x.shape[1], trn_x.shape[2])

    # output shape
    output_shape = trn_pos.shape[1]

    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, activation="relu"))
    model.add(Dense(output_shape, activation="softmax"))

    before_filepath = "best_lstm_before_retraining.hdf5"
    fitted_model = training_phase_both_models(trn_x, trn_pos, tst_x, tst_pos, model)

    # best epoch
    loss_hist = fitted_model.history['loss']
    best_epoch = np.argmin(loss_hist)
    print(best_epoch)

    # file to save weights of callback for model after retraining

    y_pred = retrain_models(trn_x, trn_pos, tst_x, tst_pos, model, best_epoch)
    y_pred = labeler.inverse_transform(y_pred)

    # print macro averaged precision, recall, f1-score and confusion matrix on test-set
    precision_recall_f1_confusionMatrix(test_pos, y_pred)



if __name__ == '__main__':
    # Not checked for grading,
    # but remember that before calling train_test_mlp() and
    # train_test_rnn(), you need to split the as training and test
    # set properly.

    # 5.1
    train_words, train_pos = read_data(
        '/Users/xujinghua/a5-asb1993-jinhxu/en_ewt-ud-dev.conllu')

    test_words, test_pos = read_data(
        '/Users/xujinghua/a5-asb1993-jinhxu/en_ewt-ud-test.conllu')

    # 5.2
    encoder = WordEncoder()
    encoder.fit(train_words)
    '''
    encoded_train_words = encoder.transform(train_words)
    encoded_test_words = encoder.transform(test_words)
    '''
    # 5.3
    # train_test_mlp(encoded_train_words, train_pos, encoded_test_words, test_pos)

    # 5.4
    encoded_trn_words = encoder.transform(train_words, flat=False)
    encoded_tst_words = encoder.transform(test_words, flat=False)
    train_test_rnn(encoded_trn_words, train_pos, encoded_tst_words, test_pos)
