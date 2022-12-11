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
    OPTIMIZER = 'adam'

    def __init__(self, name, lover, data_dict):
        self.name = name
        self.lover = lover
        self.words_mapping = data_dict['words_mapping']
        self.characters = self.words_mapping['characters']
        self.n_to_char = self.words_mapping['n_to_char']
        self.char_to_n = self.words_mapping['char_to_n']
        self.model = self.generate_model()

    def generate_model(self):
        love_model = build_model(batch_sz = 1, encoding_dimension=[len(self.n_to_char), self.EMBED_OUT], hidden_units=self.HIDDEN_UNITS, optimizer=self.OPTIMIZER)
        love_model.load_weights(str(self.MODEL_PATH))
        return love_model
        
    def random_sentence(corpus, min_seq=64, max_seq=128):
        doc = np.random.choice(corpus, 1)[0]
        text = doc['corpus']
        text_lines = text.split('\n')
        length = 0
        i=0
        sequence = str()
        while length <= min_seq:
            sequence = sequence + text_lines[i] + '\n'
            length = len(sequence)
            i+=1
        if length > max_seq:
            sequence = sequence[:max_seq]
        return text, sequence

    def predict_next_char(self, sequence, max_seq=128, creativity=3):
        if len(sequence)==0:
            sequence = '\n'
        sequence = sequence[max(0, len(sequence)-max_seq):].lower() 
        sequence_encoded = np.array([self.char_to_n[char] for char in sequence])
        sequence_encoded = np.reshape(sequence_encoded, (1, len(sequence_encoded)))
        pred_encoded = self.model(sequence_encoded)  
        pred = pred_encoded[0][-1]
        pred_prob = np.exp(pred)
        pred_prob = np.exp(pred)/(np.sum(np.exp(pred))*1.0001)
        pred_char = np.random.multinomial(creativity, np.append(pred_prob, .0))
        chars_max = pred_char==pred_char.max()
        chars_max_idx = [i for i in range(len(pred_char)) if chars_max[i]]
        char_idx = np.random.choice(chars_max_idx, 1)[0]
        if char_idx > len(self.n_to_char)-1: char_idx = ''
        char =self.n_to_char[char_idx]
        return char

    def write_poem(self, seed, max_seq=128, max_words=150, creativity=3):
        poem = seed
        print(poem, end ="")
        final = True
        word_counter = len(re.findall(r'\w+', poem)) < max_words
        while (word_counter & final):    
            next_char = self.predict_next_char(poem, max_seq, creativity)
            print(next_char, end ="")
            poem = poem + next_char
            final = poem[-1] != '$'
            word_counter = len(re.findall(r'\w+', poem)) < max_words        
        signature = '\n\nAI.S.P\n\n'
        print(signature, end ="")
        poem = poem + signature
        return poem

    
