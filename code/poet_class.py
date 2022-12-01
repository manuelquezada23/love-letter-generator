import model

class Poet:
    WEIGHT_PATH = "model/weights.h5"
    EMBED_OUT = 128
    HIDDEN_UNITS = [512, 512, 512, 512, 512]
    embedding_to_char = {"a": 1} # TODO: THIS IS JUST A PLACEHOLDER 

    def __init__(self, name, lover):
        self.model = model.LoveLetterGeneratorModel(batch_sz = 1, 
                    encoding_dimension = [len(self.embedding_to_char), self.EMBED_OUT],
                    hidden_units = self.HIDDEN_UNITS,
                    optimizer= 'adam')
        # self.model.load_weights(self.WEIGHT_PATH) TODO: UNCOMMENT THIS
        self.name = name
        self.lover = lover
    
    def write_poem(self, seed):
        # TODO: Use love letter generator model to write poem
        return f"{self.name} loves {self.lover}"