import tensorflow as tf
import numpy as np
from functools import reduce
import os
import re

"""
reads poem corpus
"""
def read_poems(data_path):
    poems = os.listdir(data_path)
    print('Processing:', len(poems), 'files\n')

    poem_data = []
    ## read through all files
    for poem in poems:
      ## with open(path + file, "rb") as train_file:
        poem_file_path = data_path + poem
        with open(poem_file_path, "rb") as poem_file:
            file = poem_file.read()
            file = file.decode('utf8')
            # lower case
            file = file.lower()
            # removes leading, ending and duplicates whitespaces
            # duplicate whitespaces especially important since some poems might be wacky
            file = re.sub(' +', ' ', file).strip()
            poem_data.append({'file': poem_file_path,
                           'corpus': file})
    return poem_data



def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg word->index mapping)
    """

    


    # read all .txt poems files
    poem_dir_path = 'data/'
    poem_corpus = read_poems(poem_dir_path)

    corpus_train, corpus_test = corpus_split(poem_corpus, split=SPLIT)

    train_x, train_y = build_data(corpus_train, char_to_n, 
                              max_seq = MAX_SEQ, stride=STRIDE)

    # # If it is a test mode just take first 25 songs
    # if TEST_MODE:
    #     print("TEST MODE: ON")
    #     corpus = corpus[:250]
    # else:
    #     print("TEST MODE: OFF")


    
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    ## TODO: Implement pre-processing for the data files. See notebook for help on this.
    with open(train_file) as f1:
        train_data = f1.readlines()
    with open(test_file) as f2:
        test_data = f2.readlines()

    train_data = ' '.join(train_data)
    # print("train_data", train_data)
    train_data = train_data.split()

    test_data = ' '.join(test_data)
    test_data = test_data.split()

    ## get unique words
    unique_words = sorted(set(train_data))
    vocabulary = {w:i for i, w in enumerate(unique_words)}
    
    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print("train_data", train_data)
    return train_data, test_data, vocabulary
