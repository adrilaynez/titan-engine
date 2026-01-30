import math
import random
from micrograd.engine import Value


# I add Module to be closer to how pytorch works and nonlil to work with relu that is better for training

# All other classes inherit from Molude

class Module: 
    
    def parameters(self):
        return  []
    
    # For training 
    def zero_grad(self): 
        for w in self.parameters(): 
            w.grad = 0.0 


class Neuron(Module): 
    def __init__(self, nin, nonlin= True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]      #Define the weights randomly
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin
    
    def __call__(self, x):
        # act = w * x + b 
        act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)    # zip the w and x => create a list [(x1,w1), .... , (xn,wn)] and sum each xi*wi multiplication (is a vector multiplication) starting from the bias
        out = act.relu() if self.nonlin else act
        # out = act.tanh() the tanh works better than relu alone (relu needs a more complex loss function but the result is a lot better)
        return out
    
    # For the training
    def parameters(self):
        return self.w + [self.b]
    
    # Copy from Andrej Karpathy code, is just to represent cleaner the neurons
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    

# List of neurons
class Layer(Module):
    def __init__ (self, nin, nout, nonlin =True ):
        self.layer = [ Neuron(nin,nonlin) for _ in range (nout)]   # You can start a class like Neuron (nin)

    def __call__(self, x):
        outs = [n(x) for n in self.layer ]
        # return outs    
        return outs[0] if len(outs) == 1 else outs      #  If there's only one output, it returns a number.   
    
    def parameters(self):
        parameters = []
        for n in self.layer:
            parameters.extend(n.parameters())
        return parameters
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.layer)}]"
    

#List of layers
class MLP(Module): 
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    
    def __call__(self, x):
        # First, add the input to the first layer. Then, use that result to add the input to the second layer. 
        for layer in self.layers:   
            x = layer(x)
        return x
    
    def parameters(self):
        parameters = []
        for l in self.layers:
            parameters.extend(l.parameters())
        return parameters
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
