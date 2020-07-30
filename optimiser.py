import numpy as np


class Optimiser:
    def __init__(self, algorithm, loss_function, parameters, learning_rate, epochs):
        self.algorithm = algorithm
        self.loss_function = loss_function
        self.parameters = parameters
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        
    def gradient_descent(self)
        pass
        
    def minimise(self):
    
        if self.algorithm == "GD":
            return self.gradient_descent(self.epochs)
        else:
            print("Choose a valid optimiser\n")
            print("Available optimisers:\n 1. Standard Gradient Descent")
