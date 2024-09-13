from rsdl.optim import Optimizer
import numpy as np

# TODO: implement Adam optimizer like SGD
class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.1, momentum = 0.01, b1=0.9, b2=0.999, epsilon=1):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.b1=b1
        self.b2=b2
        self.m=0.0
        self.v=0.0
        self.epsilon = epsilon

    def step(self):
        # TODO: update weight and biases ( Don't use '-=' and use l.weight = l.weight - ... )
        for l in self.layers:
            self.m = self.b1 * self.m + (1 - self.b1) * [x.grad for x in l.weight]
            self.v = self.b2 * self.v + (1 - self.b2) * ([x.grad for x in l.weight] ** 2)
            l.weight = l.weight - (self.learning_rate * self.m / np.sqrt(self.v + self.epsilon))
            if l.need_bias:
                l.bias = l.bias - self.learning_rate * l.bias.grad