import pickle
from model import LoveLetterGeneratorModel
from tensorflow.keras.callbacks import ModelCheckpoint

EPOCHS = 30
BATCH_SIZE = 256
ENCODING_OUT = 128
HIDDEN_UNITS = [512, 512, 512, 512, 512]
CORPUS_PATH = '../data/processed/processed_poems'

with open(CORPUS_PATH + '.pickle', 'rb') as file:
    data = pickle.load(file)

corpus = data['corpus']
words_mapping = data['words_mapping']
train_x = data['train_x']
train_y = data['train_y']
test_x = data['test_x']
test_y = data['test_y']

characters = words_mapping['characters']
n_to_char = words_mapping['n_to_char']
char_to_n = words_mapping['char_to_n']

MODEL_PATH = './models/'
MODEL_NAME = 'love-letter-generator-model'

LoveLetterGeneratorModel = LoveLetterGeneratorModel(batch_sz = BATCH_SIZE, 
                    encoding_dimension = [len(n_to_char), ENCODING_OUT],
                    hidden_units = HIDDEN_UNITS,
                    optimizer= 'adam')

model = LoveLetterGeneratorModel.model

sample_train = (len(train_x)//BATCH_SIZE)*BATCH_SIZE

checkpoint = ModelCheckpoint(str(MODEL_PATH + MODEL_NAME + '.h5'),
                             verbose=1, period=3)

model_history = model.fit(train_x[:sample_train,:], train_y[:sample_train,:],
                          epochs = EPOCHS,
                          batch_size = BATCH_SIZE,
                          callbacks = [checkpoint]) 
