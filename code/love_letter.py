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

    POEMS_PATH = 'past_poems/'
    poet_name = input('What is the name of the poet?\n')
    lover = input('Who is the letter dedicated to?\n')

    poet = Poet(poet_name, lover, data_dict)
    
    while True:
        try:
            seed = input('Feeling inspired? How do you want to start the poem\n')
            # write poem
            poem = poet.write_poem(seed)
            print(poem)
            # save poem
            filename = convert_to_filename(poem.split('\n')[0])
            with open(POEMS_PATH + filename + '.txt', "w") as text_file:
                text_file.write(poem)
            print('Poem saved as: ' + filename + '.txt')
            repeat = input('Another poem? (Y/N) ')
            if repeat.lower() == 'n':
                break
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
