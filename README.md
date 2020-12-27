# [Pos tagging with ANNs](https://snlp2020.github.io/a5/)

Pos tagging with feed-forward (MLP)
and recurrent network architectures.
A simplified POS tagging problem.
We are interested in predicting the POS tag of 
a word using only the characters (or character sequences) present in the word.
In a real-world POS tagging application,
it is almost unthinkable not to use the context of the word.

Data comes from
[Universal Dependencies](https://universaldependencies.org/) treebanks.
[CoNLL-U](https://universaldependencies.org/format.html)-formatted treebank.


Library:
[Keras](https://www.tensorflow.org/api_docs/python/tf/keras)


### 5.1 Read the data
### 5.2 Encoding words
### 5.3 Training and testing an MLP 
### 5.4 Training and testing an RNN 
