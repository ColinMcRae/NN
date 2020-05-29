import net1 as network
from mlxtend.data import loadlocal_mnist
import numpy as np
import time 
from numba import jit

def vectorized_result(j):    
  e = np.zeros(10)
  e[j] = 1.0
  return e

def normalize_input(X):
  return np.array([x / 255 for x in X ])

images, labels = loadlocal_mnist(
        images_path='../Datasets/MNIST_digits/train-images-idx3-ubyte', 
        labels_path='../Datasets/MNIST_digits/train-labels-idx1-ubyte')

#net = network.Network([784, 64, 40, 10])
net = network.load('MNIST_digits', 20)

#images = images[:10000]
#labels = labels[:10000]

for ep in range(10, 100):
  example = 0
  ep_loss = 0
  start = time.time()
  for X, y in zip(images, labels):
    ep_loss += net.train(normalize_input(X), vectorized_result(y), ep)
    #ep_loss += network.mse_loss(vectorized_result(y), net.feedforward(X))
    example += 1
    if example % 10000 == 0:      
      end = time.time()
      if example > 0:
      	print(ep_loss / example, ' in', end - start)
      	ep_loss = 0
      start = time.time()
  if ep %2 == 0:
  	net.save('MNIST_digits', ep)