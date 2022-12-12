import pickle
import my_model
from tensorflow.keras.callbacks import ModelCheckpoint

EPOCHS = 30
BATCH_SIZE = 256
ENCODING_OUT = 128
HIDDEN_UNITS = [512, 512, 512, 512, 512]
CORPUS_PATH = '../data/processed/processed_poems'

"""
creates weights and stores them in models. this is already done by us, please look at README
"""
def train():
    with open(CORPUS_PATH + '.pickle', 'rb') as file:
        data = pickle.load(file)

    words_mapping = data['words_mapping']
    train_x = data['train_x']
    train_y = data['train_y']

    n_to_char = words_mapping['n_to_char']

    MODEL_PATH = './models/'
    MODEL_NAME = 'love-letter-generator-model'

    love_letter_model = my_model.build_model(batch_sz = BATCH_SIZE, 
                    encoding_dimension = [len(n_to_char), ENCODING_OUT],
                    hidden_units = HIDDEN_UNITS,
                    optimizer = 'adam')

    sample_train = (len(train_x)//BATCH_SIZE)*BATCH_SIZE


    checkpoint = ModelCheckpoint(str(MODEL_PATH + MODEL_NAME + '.h5'),
                             verbose=1, period=1)

    love_letter_model.fit(train_x[:sample_train,:], train_y[:sample_train,:],
                          epochs = EPOCHS,
                          batch_size = BATCH_SIZE,
                          callbacks = [checkpoint])

    love_letter_model.save(str(MODEL_PATH + MODEL_NAME + '.h5'))

if __name__ == '__main__':
    train()