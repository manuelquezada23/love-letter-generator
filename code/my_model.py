import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Embedding
from losses import scc_loss

# model params
def build_model(batch_sz, encoding_dimension, hidden_units, optimizer, loss=scc_loss):
    model = Sequential()
    model.add(Embedding(input_dim = encoding_dimension[0], output_dim = encoding_dimension[1],
                        batch_input_shape = [batch_sz, None]))
    for units in hidden_units:
        model.add(LSTM(units, return_sequences=True, recurrent_initializer='glorot_uniform',
                   dropout=0.25, recurrent_dropout=0.05))
    model.add(Dense(encoding_dimension[0]))
    model.compile(loss = loss,
                  optimizer = optimizer)
    return model



