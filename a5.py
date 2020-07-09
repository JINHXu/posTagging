#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 5
    See <https://snlp2020.github.io/a5/> for detailed instructions.

    <Please insert your name and the honor code here.>
"""
import random
from _csv import reader

import numpy as np
import nltk
import pandas as pd
import tf as tf


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
        print(long_word, longest_word)

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
        print(self._singleChars)

        self._total_number_chars = len(self._singleChars)
        print(self._total_number_chars)

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

        for word in words:
            # truncate if word longer than maxlen
            if len(word) > self._maxlen:
                word = word[:self._maxlen]
            # for each character in each word
            for char in word:
                # create empty array of zeros
                word_char_vec = np.zeros((self._total_number_chars,), dtype=int)
                # if char of word equals character in singleChar, add 1 at the specified position of this char determined by the dictionary _singleChar
                if char in self._singleChars:
                    word_char_vec[self._singleChars[char]] = 1
                # else if char not in _singleChar, then it is UNK and therefore, at position two will be a 1
                else:
                    char = "UNK"
                    word_char_vec[self._singleChars[char]] = 1
            if word < self._maxlen:
                if pad is "right":
                    word






        #vector_char_words = [[0 if char != letter selse 1 for char in self._singleChars] for letter in self._word_list]


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


if __name__ == '__main__':
    ### Not checked for grading,
    ### but remember that before calling train_test_mlp() and 
    ### train_test_rnn(), you need to split the as training and test
    ### set properly.

    #
    # dataset downloaded from https://github.com/UniversalDependencies/UD_English-ParTUT
    words, pos_tags = read_data("en_partut-ud-dev.conllu")
    encoder = WordEncoder()
    encoder.fit(words)
    encoder.transform(words)
