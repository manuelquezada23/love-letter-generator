import numpy as np
import pickle
from poet_class import Poet
from preprocess import get_data
import re
from my_model import build_model

def convert_to_filename(value):
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value

def main():
    get_data()

    DATA_PATH = '../data/processed/processed_poems.pickle'
    with open(DATA_PATH, 'rb') as data_file:
        data_dict = pickle.load(data_file)

    POEMS_PATH = 'code/'
    poet_name = input('What is the name of the poet writing?\n')
    recipient = input('Who is the poem/letter dedicated to?\n')

    poet = Poet(poet_name, recipient, data_dict)
    
    while True:
        try:
            seed = input('Feeling inspired? How do you want to start the poem/letter?\n')
            print("Writing...")
            # write poem
            poem = poet.create_poem(seed)
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
