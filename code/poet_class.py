import model
import re
import np

class Poet:
    WEIGHT_PATH = "model/weights.h5"
    EMBED_OUT = 128
    HIDDEN_UNITS = [512, 512, 512, 512, 512]

    def __init__(self, name, lover, data_dict, weight_path = WEIGHT_PATH):
        self.model = model.LoveLetterGeneratorModel(batch_sz = 1, 
                    encoding_dimension = [len(self.embedding_to_char), self.EMBED_OUT],
                    hidden_units = self.HIDDEN_UNITS,
                    optimizer= 'adam')
        self.model.load_weights(self.WEIGHT_PATH) 
        self.name = name
        self.lover = lover
        words_mapping = data_dict['words_mapping']
        characters = words_mapping['characters']
        n_to_char = words_mapping['n_to_char']
        char_to_n = words_mapping['char_to_n']

        
    def random_sentence(corpus, min_seq=64, max_seq=128):
        # use random sentence from corpus as seed
        doc = np.random.choice(corpus, 1)[0]
        text = doc['corpus']
        text_lines = text.split('\n')
        # select first lines till min_seq constraint is reached
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
        '''
        sequence: input sequence seen so far. (if blank model will start with a random character)
        max_seq: maximum number of characters to seen in one sequence (use the same sequence as model)
        creativity: 1: super creative, 10: Conservative.
        '''
        # start with a random character
        if len(sequence)==0:
            sequence = '\n'
        # cut sequence into max length allowed   
        sequence = sequence[max(0, len(sequence)-max_seq):].lower() 
        # transform sentence to numeric
        sequence_encoded = np.array([self.char_to_n[char] for char in sequence])
        # reshape for single batch predicton
        sequence_encoded = np.reshape(sequence_encoded, (1, len(sequence_encoded)))
        # model prediction
        pred_encoded = model.predict(sequence_encoded)  
        # last prediction in sequence
        pred = pred_encoded[0][-1]
        # from log probabilities to normalized probabilities
        pred_prob = np.exp(pred)
        pred_prob = np.exp(pred)/(np.sum(np.exp(pred))*1.0001)
        # get index of character  based on probabilities
        # add an extra digit (issue from np.random.multinomial)
        pred_char = np.random.multinomial(creativity, np.append(pred_prob, .0))
        # character with highest aperances
        chars_max = pred_char==pred_char.max()
        # get index of those characters
        chars_max_idx = [i for i in range(len(pred_char)) if chars_max[i]]
        char_idx = np.random.choice(chars_max_idx, 1)[0]
        # if prediction do not match vocabulary. do nothing
        if char_idx > len(self.n_to_char)-1: char_idx = ''
        char =self.n_to_char[char_idx]
        return char

    def write_poem(self, seed, max_seq=128, max_words=150, creativity=3):
        # start poem with the seed
        poem = seed
        print(poem, end ="")
        # placeholder stopers 
        final = True
        word_counter = len(re.findall(r'\w+', poem)) < max_words
        # ends poem generator if max word is passed or poem end with final dot ($)
        while (word_counter & final):    
            # Prediction next character
            next_char = self.predict_next_char(poem, max_seq, creativity)
            print(next_char, end ="")
            # append
            poem = poem + next_char
            # update stopers
            final = poem[-1] != '$'
            word_counter = len(re.findall(r'\w+', poem)) < max_words        
        # add signature
        signature = '\n\nAI.S.P\n\n'
        print(signature, end ="")
        poem = poem + signature
        return poem


    