import numpy as np
import pickle

def main(args):
    DATA_PATH = './data/'
    with open(DATA_PATH, 'rb') as data_file:
        data_dict = pickle.load(data_file)
    
    model = LoveLetterGenerator()

if __name__ == '__main__':
    main()
