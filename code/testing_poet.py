import poet_class
import pickle

DATA_PATH = '../data/processed/processed_poems.pickle'
with open(DATA_PATH, 'rb') as data_file:
    data_dict = pickle.load(data_file)

poet = poet_class.Poet('Shakespeare', 'Juliet', data_dict)

poem = poet.write_poem('hearts to respond')