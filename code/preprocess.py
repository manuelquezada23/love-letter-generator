import tensorflow as tf
import numpy as np
import pandas as pd
import nltk
from functools import reduce
import os
import re
import pickle

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

def clean_data():
    ############ Character per doc.
    chars = [len(x['corpus']) for x in corpus]
    print('\nCharacters per doc:\n', pd.Series(chars).describe())
    # remove outliers, less than 260 characters
    idx = [x > 260 for x in chars]
    print('docs removed:', len(corpus) - sum(idx))
    corpus = list(np.array(corpus)[np.array(idx)])


    ############ lines per doc, remove single liners
    lines = [len(x['corpus'].split('\n')) for x in corpus]
    print('\nLines per doc:\n', pd.Series(lines).describe())
    # remove outliers. less than 10 lines
    idx = [((x > 8) & (x < 50)) for x in lines]
    print('docs removed:', len(corpus) - sum(idx))
    corpus = list(np.array(corpus)[np.array(idx)])


    ############ words count.
    words = [len(re.findall(r'\w+', x['corpus'])) for x in corpus]
    print('\nwords per doc:\n', pd.Series(words).describe())
    # remove outliers
    # less than 100 words
    idx = [((x > 30) & (x < 200)) for x in words]
    print('docs removed:', len(corpus) - sum(idx))
    corpus = list(np.array(corpus)[np.array(idx)])

    words = [re.findall(r'\w+', x['corpus']) for x in corpus]
    # flat list into single array to count frequency
    words = [item for sublist in words for item in sublist]

    print('\nTotal words in corpus', len(words))
    print('\nUnique words:', len(set(words)))

    # most common words couont
    words_counter = nltk.FreqDist(words)
    words_counter = pd.DataFrame(words_counter.most_common())
    words_counter.columns = ['words', 'freq']
    words_counter['len'] = [len(x) for x in words_counter['words']]
    print('\Most common words:', '\nat least 4 digits:\n',
          words_counter[['words','freq']][words_counter['len']>=4].head(10),
          '\nat least 6 digits:\n',
          words_counter[['words','freq']][words_counter['len']>=6].head(10))
    

    ##### Clean less frecuent characters
    # want all docs as a single string to get unique chars
    all_text = str()
    for x in corpus:
        all_text = all_text + x['corpus']
  
    # unique characters
    characters = sorted(list(set(all_text)))
    print('unique characters:', len(characters))

    # count of characters
    print('Number of appereance per unique character in corpus')
    chars_count = []
    for digit in characters:
        counter = {'digit': digit,
                            'count': sum([digit in char for char in all_text])}
        print(counter)
        chars_count.append(counter)
    chars_count = pd.DataFrame(chars_count)

    # remove non meaningfull characters
    # manually make the cut at 766
    # chars_count['digit'][chars_count['count'] <= 766].values
    chars_remove = ['\t', '"', '%', '&', "'", '\*', '\+', '/', '0', '1', '2', '3', '4',
                    '5', '6', '7', '8', '9', '=', '\[', '\]', '_',  '`', '{',
                    '¨', 'ª', '«', '²', '´', 'º', '»', 'à', 'â', 'ã', 'ä', 'ç',
                    'è', 'ê', 'ì', 'ï', 'ò', 'ô', 'õ', 'ö', 'ü', '\xa0', '°']

    print('\nErase from docs non meaningful characters:', chars_remove)
    # as reggex expression
    chars_remove = '[' + ''.join(chars_remove) + ']'
    # erase from docs non meaninful characters
    for doc in corpus:
        doc['corpus'] = re.sub(chars_remove, ' ', doc['corpus'])
        doc['corpus'] = re.sub(' +', ' ', doc['corpus']).strip()

    # line space: '\r\n ' '\r\n' to '\n', '\r\r\n'
    for doc in corpus:
        for pattern in ['\r\r\n', '\r\n ', '\r\n', '\n\n']:
            doc['corpus'] = re.sub(pattern, '\n', doc['corpus'])

    # add special charater for at the begining and final of text.
    # Model will learn when to start/end
    for doc in corpus:
        doc['corpus'] = doc['corpus'] + '.$'

    return corpus


def get_vocabulary(corpus):
    '''
    Mapping is a step in which we assign an arbitrary number to a character/word
    in the text. In this way, all unique characters/words are mapped to a number.
    This is important, because machines understand numbers far better than text,
    and this subsequently makes the training process easier.
    '''
    # create dictionaries for character/number mapping
    print('\n---\nCreating word mapping dictionaries')

    # want all docs as a single string to get unique chars
    all_text = str()
    for x in corpus:
        all_text = all_text + x['corpus']
  
    # unique characters
    characters = sorted(list(set(all_text)))
    print('unique characters after cleansing', len(characters))
    print(''.join(characters))
    
    # dictionaries to be used as index mapping
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}
    
    return characters, n_to_char, char_to_n  

# =============================================================================
# split train and test
# =============================================================================
def corpus_split(corpus, split):
    # Train/Test per doc
    # create corpus index
    idx = [i for i in range(len(corpus))]
    # random random for train
    idx_train = np.random.choice(idx, size=int(len(corpus)*split), replace=False)
    # index not in train
    idx_test = [i for i in idx if i not in idx_train]
    # split corpus by index
    corpus_train = [corpus[i] for i in idx_train]
    corpus_test = [corpus[i] for i in idx_test]    
    # Docs stats corpus
    print('\n--- Total docs in corpus', len(corpus))
    print('split in:')
    print('Train:', len(corpus_train))
    print('Test:', len(corpus_test))
    return corpus_train, corpus_test

# Create tensor data from corpus
def build_data(corpus, char_to_n, max_seq = 100, stride = [1,6]):
    '''
    Transform list of documents into tensor format to be fed to a lstm network
    outout shape: (sequences, max_lenght)
    max_seq: maximum length of sequence
    stride: steps apply in rolling window over text. next windows could be next character(1) or 6 characters ahead
    '''
    # place holder to  save results
    data_x = []
    data_y = []
    sequences = []
    # target sequence is lagged by 1, hence sequence length = max_seq+1
    max_seq+=1  
    # for each document in corpus
    for i in range(len(corpus)):
        if (i % max(1, int(len(corpus)/10)) == 0):
            print('\n--- Progress %:{0:.2f}'.format(i/len(corpus)))
        text = corpus[i]['corpus']
        text_length = len(text)
        # iterate for all text in rolling windows of size max_seq
        j = max_seq
        while j < text_length + stride[0]:
            k_to = min(j, text_length) # 
            k_from = (k_to - max_seq)            
            #print(j, ':', k_from, '-', k_to)
            #♠ slice text
            sequence = text[k_from:k_to] 
            # characters to int
            sequence_encoded = np.array([char_to_n[x] for x in sequence]) 
            # append results
            sequences.append(sequence)
            data_x.append(sequence_encoded[:-1])
            data_y.append(sequence_encoded[1:])
            # random stride between 1-6
            j+=int(np.random.uniform(stride[0], stride[1]+1))       
    # Tensor structure
    data_x = np.array(data_x) 
    data_y = np.array(data_y) 
    
    # Shuffle data
    data_x = tf.random.shuffle(data_x)
    data_y = tf.random.shuffle(data_y)

    # output
    print('Outupt shape -', 'X:', data_x.shape, '- Y:', data_y.shape)
    size = data_x.nbytes*1e-6 + data_y.nbytes*1e-6
    size = print(int(size), 'Megabytes')
    return data_x, data_y

def get_tensor_data(corpus_train, corpus_test, char_to_n, max_seq, stride ):
  # Create tensor data from corpus
  print('\n---\nBuild Tensor data')
  # Train datasets
  print('\nBuild Train data:')
  train_x, train_y = build_data(corpus_train, char_to_n, 
                                max_seq = max_seq, stride=stride)

  if len(corpus_test):
      print('\nBuild Test data:')
      test_x, test_y = build_data(corpus_test, char_to_n, 
                                    max_seq = max_seq, stride=stride)
  else:
      test_x, test_y = None, None
  
  return train_x, train_y, test_x, test_y

def get_data():
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
    # is it a test? if True only 25 songs are used
    TEST_MODE = False

    # Split train/test
    SPLIT = 1

    # NLP
    MAX_SEQ = 120 # sequence length
    STRIDE = [MAX_SEQ/2, MAX_SEQ] 

    # output path
    OUTPUT_PATH = './data/data_processed/'
    # OUTPUT_FILE = 'NLP_data_poems_120_no-split'

    # read all .txt poems files
    poem_dir_path = '../poetry_data/'
    poem_corpus = read_poems(poem_dir_path)

    corpus = clean_data()

    characters, n_to_char, char_to_n =  get_vocabulary(corpus)

    # mapping used as part of the model
    words_mapping = {'characters': characters,
                    'n_to_char': n_to_char,
                    'char_to_n': char_to_n}




    corpus_train, corpus_test = corpus_split(poem_corpus, split=SPLIT)

    train_x, train_y, test_x, test_y = get_tensor_data(corpus_train=corpus_train, corpus_test=corpus_test, max_seq=MAX_SEQ, stride=STRIDE)

    # Merge all data in one dict
    data = {'corpus': corpus,
            'words_mapping': words_mapping,
            'train_x': train_x ,
            'train_y': train_y,
            'test_x' : test_x,
            'test_y' : test_y}

    # save file
    with open('data/processed/processed_poems.pickle', 'wb') as file:
        pickle.dump(data, file)
    print('Data saved in:', 'data/processed/processed_poems.pickle')
  

    return corpus_train, corpus_test

    # want to end with this:
    # return train_data, test_data, vocabulary

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
