import numpy as np


class Neuron:
    def __init__(self, activation, input_vector, output_dimension)
    
        self.activation = activation
        self.input_dimension = input_vector.shape
        self.output_dimension = output_dimension
        
        self.X = input_vector
        self.Weights = np.ones((self.input_dimension, self.output_dimension))
        self.Bias = np.zeros((1,self.output_dimension))
        
        self.Output = np.zeros(output_dimension)
        
    def perceptron(self):
        pass
        
    def sigmoid_neuron(self):
        pass
        
    def activated_neuron(self):
        pass
    
        
        
        

