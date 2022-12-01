from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy

class LoveLetterGeneratorModel(tf.keras.Model):
    def scc_loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

    def __init__(self, batch_sz, encoding_dimension, hidden_units, optimizer, loss=scc_loss):
        super().__init__()
        model = Sequential()
        # Add embedding layer
        model.add(Embedding(input_dim = encoding_dimension[0], output_dim = encoding_dimension[1],
                            batch_input_shape = [batch_sz, None]))
        # Add LSTM layers
        for units in hidden_units:
            model.add(LSTM(units, return_sequences=True, recurrent_initializer='glorot_uniform',
                    dropout=0.25, recurrent_dropout=0.05))
        # Add output layer
        model.add(Dense(encoding_dimension[0]))
        # Compile model
        model.compile(loss = loss,
                    optimizer = optimizer)
        self.model = model

    def call(self, x):
        return self.model(x)
    
    



