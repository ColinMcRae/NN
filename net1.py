import numpy as np
from numba import jit
import pickle
import os

def mse_loss(y_true, y_pred):  
  return ((y_true - y_pred) ** 2).mean()  

def new(layers): # creates new network
  return Network(layers)

def load(directory, epoch): # loads model from file
  filename = directory + '/weights_' + str(epoch)# maybe, open last file automatically
  weights = pickle.load( open( filename , 'rb' ))
  layers = np.array([len(x) for x in weights])
  layers = np.append(layers, len(weights[-1][0]))  
  net = Network(layers)
  net.weights = weights

  return net

class Network:
  def __init__(self, layers):
    self.layers = len(layers)
    self.weights = self.__init_weights(layers)
    self.biases = self.__init_biases(layers)
    self.learn_rate = 0.08
    self.layer_sizes = layers
    self.neurons = self.__init_neurons(layers)

  def __init_weights(self, layers):
    weights = []   
    for i in range(len(layers) - 1):
      weights.append(np.random.normal(0.0, 1, (layers[i], layers[i + 1])))  
    return weights

  def __init_biases(self, layers):
    biases = []
    for i in range(len(layers) - 1):
      bias = [0] * layers[i + 1]
      biases.append(bias)
    return biases  

  def __init_neurons(self, layers):
    neurons = []
    for layer in layers:
      neurons.append(np.zeros(layer))
    return neurons  

  def save(self, directory, epoch):
    filename = directory + '/weights_' + str(epoch)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pickle.dump( self.weights , open( filename , 'wb' ) )

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))  

  def sigmoid_der(self, x):
    return x * (1 - x)   

  def mse_loss(y_true, y_pred):  
    return ((y_true - y_pred) ** 2).mean()    

  def feedforward(self, inputs):    
    for i in range(self.layers - 1):
      inputs = np.dot(inputs, self.weights[i]) + self.biases[i]               
      outputs = np.array([self.sigmoid(x) for x in inputs])
      inputs = outputs    
    return outputs

  def back(self, inputs):
    #TODO synthesis data from classified input
    return

  def train(self, inputs, excepted_out, epoch):
    outputs = []    
    outputs.append(inputs)

    for i in range(self.layers - 1):
      inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
      outputs.append(np.array([self.sigmoid(x) for x in inputs]))
      inputs = outputs[-1]
    
    errors = np.array(outputs[-1]) - np.array(excepted_out)
    loss = (errors ** 2).mean()    
        
    for layer in reversed(range(0, self.layers - 1)):    
      weights_delta = np.array([self.sigmoid_der(x) for x in outputs[layer + 1]]) * errors                
      self.weights[layer] -= np.dot(np.array([outputs[layer]]).T, np.array([weights_delta])) * self.learn_rate      
      self.biases[layer] -= np.dot(np.multiply(weights_delta, outputs[layer + 1]), self.learn_rate)      
      errors = np.dot(self.weights[layer], weights_delta)      

    return loss  

  @jit(nopython=True) # TODO  
  def train_nopython(self, inputs, excepted_out, epoch):
    for i in range(self.layers - 1):
      inputs = np.dot(inputs, self.weights[i]) + self.biases[i]
      act_inputs = np.array([self.sigmoid(x) for x in inputs])
      self.neurons[i] = inputs = act_inputs      

    errors = np.array(self.neurons[-1]) - np.array(excepted_out)
    layer = self.layers - 1    
    while layer >= 0:  
      weights_delta = np.array([self.sigmoid_der(x) for x in self.neurons[layer + 1]]) * errors
      self.weights[layer] -= np.dot(np.array([self.neurons[layer]]).T, np.array([weights_delta])) * self.learn_rate      
      errors = np.dot(self.weights[layer], weights_delta)
      layer =-1
