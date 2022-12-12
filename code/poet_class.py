from my_model import build_model
import re
import numpy as np
import tensorflow as tf
import os
from losses import scc_loss

class Poet:
    MODEL_PATH = './models/love-letter-generator-model.h5'
    EMBED_OUT = 128
    HIDDEN_UNITS = [512, 512, 512, 512, 512]

    def __init__(self, name, recipient, data_dict):
        self.name = name
        self.recipient = recipient
        self.words_mapping = data_dict['words_mapping']
        self.characters = self.words_mapping['characters']
        self.n_to_char = self.words_mapping['n_to_char']
        self.char_to_n = self.words_mapping['char_to_n']
        self.model = self.generate()

    def generate(self):
        love_model = build_model(batch_sz = 1, encoding_dimension=[len(self.n_to_char), self.EMBED_OUT], hidden_units=self.HIDDEN_UNITS, optimizer='adam')
        love_model.load_weights(str(self.MODEL_PATH))
        return love_model
        
    def random_sentence(corpus, minimum=64, maximum=128):
        text = np.random.choice(corpus, 1)[0]['corpus']
        split_text = text.split('\n')
        text_length = 0
        counter = 0
        sequence = str()

        while text_length <= minimum:
            sequence = sequence + split_text[counter] + '\n'
            text_length = len(sequence)
            counter += 1

        if text_length > maximum:
            sequence = sequence[:maximum]

        return text, sequence

    def predict_character(self, seq, maximum=128, creativity=3):

        if len(seq)==0:
            seq = '\n'

        seq = seq[max(0, len(seq)-maximum):].lower() 
        encoded = np.array([self.char_to_n[char] for char in seq])
        encoded = np.reshape(encoded, (1, len(encoded)))
        
        pred = self.model(encoded)  
        pred_prob = np.exp(pred[0][-1])
        pred_prob = np.exp(pred[0][-1])/(np.sum(np.exp(pred[0][-1]))*1.0001)
        pred_char = np.random.multinomial(creativity, np.append(pred_prob, .0))
        chars_max = pred_char == pred_char.max()

        chars = []
        for i in range(len(pred_char)):
            if chars_max[i]:
                chars.append(i)

        index = np.random.choice(chars, 1)[0]
        if index > len(self.n_to_char) - 1:
            index = ''

        char = self.n_to_char[index]
        return char

    def create_poem(self, inspiration, maximum_sequence=128, maximum_words=150, creativity=3):
        res = inspiration
        bool = True
        counter = len(re.findall(r'\w+', res)) < maximum_words
        print("\n\nDear " + self.recipient + ",")
        while (counter & bool):    
            next_char = self.predict_character(res, maximum_sequence, creativity)
            print(next_char, end ="")
            res = res + next_char
            bool = res[-1] != '$'
            counter = len(re.findall(r'\w+', res)) < maximum_words        
        signature = 'With love,\n ' + self.name + '\n'
        print(signature, end ="")
        res = res + signature
        return res

    
