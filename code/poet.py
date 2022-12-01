import model

class Poet:
    WEIGHT_PATH = "model/weights.h5"
    def __init__(self, name, lover):
        self.model = model.LoveLetterGeneratorModel()
        self.model.load_weights(self.WEIGHT_PATH)
        self.name = name
        self.lover = lover
    
    def write_poem(self):
        # TODO: Use love letter generator model to write poem
        return f"{self.name} loves {self.lover}"