from tensorflow.keras.losses import sparse_categorical_crossentropy

def scc_loss(y_true, y_pred):
        return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)