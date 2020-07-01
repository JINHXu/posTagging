#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    <Please insert your name and the honor code here.>
"""

import random


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
        self._char2idx['<s>'] = 1
        self._char2idx['</s>'] = 2
        # reserve for unknown chararacters
        self._char2idx['uk'] = 3

        # current index
        idx = 4
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


'''
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
    ### 5.4
    '''


if __name__ == '__main__':
    # Not checked for grading,
    # but remember that before calling train_test_mlp() and
    # train_test_rnn(), you need to split the as training and test
    # set properly.

    # 5.1
    words, pos = read_data(
        '/Users/xujinghua/a5-asb1993-jinhxu/en_ewt-ud-dev.conllu')

    # 5.2
    encoder = WordEncoder()
    encoder.fit(words)
