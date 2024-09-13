from rsdl.optim import Optimizer

# TODO: implement Momentum optimizer like SGD
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum = 0.01):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.change = 0.0
        self.momentum = momentum

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            self.change = self.learning_rate * [x.grad for x in l.weight] + self.momentum * self.change
            l.weight = l.weight - self.change
            if l.need_bias:
                l.bias = l.bias - self.learning_rate * l.bias.grad