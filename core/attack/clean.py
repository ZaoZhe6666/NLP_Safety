from torch import nn

class CLEAN:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def attack(self, x, y):

        return x