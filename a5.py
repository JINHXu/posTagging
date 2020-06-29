#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    <Please insert your name and the honor code here.>
"""

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


if __name__ == '__main__':
    ### Not checked for grading,
    ### but remember that before calling train_test_mlp() and 
    ### train_test_rnn(), you need to split the as training and test
    ### set properly.
