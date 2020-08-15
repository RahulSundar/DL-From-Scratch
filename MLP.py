import numpy as np
#import matplotlib.pyplot as plt


'''This is a code to implement a MLP using just numpy to model XOR logic gate.Through this code, one can hope to completely unbox how a MLP model is setup.'''

# Model Parameters
'''This should consist of the no. of input, output, hidden layer units. Also, no.of inputs and outputs. '''
N_HiddenLayers = 1
Total_layers = 3
N_training = 4
Ni_units = 2
Nh_units = 2
No_units = 1

#Training dataset
'''Here we also define training examples.'''
#input arrays
x = np.zeros((N_training, Ni_units))
x[:,0] = [0,1,0,1]
x[:,1] = [0,1,1,0]
# Target Values
target = np.zeros((N_training, No_units))
target[:,0] = [0,0,1,1]
#hidden layer 
h = np.zeros((N_training,Nh_units))
#Output layer
y = np.zeros((N_training, No_units))

#Weights and biases
'''No. of weight matrices/biases a Total_layers - 1 = 2'''
W1 = np.ones((Nh_units, Ni_units))/2
W2 = np.ones((No_units, Nh_units))/2

b1 = -np.ones((Nh_units, 1))*0.5
b2 = -np.ones((No_units, 1))
  
  
#Activation function:
def sigmoid(z):
	return 1/(1+np.exp(-z))
  
def binary_threshold(z):
	return np.where(z>0, 1,0)
#Model: MLP (Simplest FFNN)
'''The model is set up as sequential matrix multiplications'''

'''Activation applied only on hidden layers as of now. Input layer's activation function is identity.'''

'''pre activation for input layer'''
z1 = np.matmul(x,np.transpose(W1)) + np.matmul(np.ones((N_training,1)), np.transpose(b1))
'''Activation for hidden layer'''
h = np.where(z1 >0, 1, 0)
'''Pre - activation of output layer '''
z2 =np.matmul(x,np.transpose(W2)) + np.matmul(np.ones((N_training,1)), np.transpose(b2))
'''Activation for output layer'''
y = np.where(z1 >0, 1, 0)


#Loss Function
'''Here, we shall be using mean squared error loss as the loss function'''
J = np.sum((y-target)**2)/N_training
print(J)
