import pickle
import numpy as np
from src.rose import *

class network: 
  def __init__(self, Model): 
    self.model = Model
    self.soft_loss = SoftmaxCrossEntropy() # Loss function 
    pass

  def forward(self, input):
    for layer in self.model:
      input = layer.forward(input)
    return input

  def backward(self, input):
    for layer in reversed(self.model):
      input = layer.backward(input)
    pass

  # save weights and biases
  def save(self, name):  
    params = {}
    for i, layer in enumerate(self.model):
      if (type(layer) == Convolution or type(layer) == Linear ):
        params[i] = layer.params
    with open(name, 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pass

  # load weights and biases
  def load(self, name):
    with open(name, 'rb') as handle:
        self.params = pickle.load(handle)
    for i, layer in enumerate(self.model):
      if (type(layer) == Convolution or type(layer) == Linear ):
        layer.params = params[i] 
    pass

  def optimize(self, lr, batch_size):
    for layer in self.model:
      if (type(layer) == Convolution or type(layer) == Linear ):
        # SGD
        layer.params["b"] -= (lr/batch_size) * layer.grads["b"]
        layer.params["w"] -= (lr/batch_size) * layer.grads["w"]

        #cleaning gradients
        layer.grads["b"] = np.zeros(layer.grads["b"].shape)
        layer.grads["w"] = np.zeros(layer.grads["w"].shape)
    pass
