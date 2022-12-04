import numpy as np
import pickle
from model import LoveLetterGeneratorModel
from poet_class import Poet
import re
import os  

POEMS_PATH = 'past_poems/'
poet_name = input('What is the name of the poet?\n\n')
lover = input('Who is the letter dedicated to?\n\n')
poet = Poet(poet_name, lover)

def convert_to_filename(value):
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value

def main():
    # DATA_PATH = './data/'
    # with open(DATA_PATH, 'rb') as data_file:
    #     data_dict = pickle.load(data_file)
    
    # model = LoveLetterGeneratorModel()

    while True:
        try:
            seed = input('Feeling inspired? How do you want to start the poem\n\n')
            print(poet.write_poem(seed))
            # write poem
            poem = poet.write_poem(seed)
            # write poem as .txt in putpus
            filename = convert_to_filename(poem.split('\n')[0])
            with open(POEMS_PATH + filename + '.txt', "w") as text_file:
                text_file.write(poem)
            # another poem?
            repeat = input('Another poem? (Y/N) ')
            if repeat.lower() == 'n':
                break
        except KeyboardInterrupt:
            break

if __name__ == '__main__':
    main()
