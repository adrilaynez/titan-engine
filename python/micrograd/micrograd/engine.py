import math 

class Value: 

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)                 # He uses "_" for all the variables that should be private but that thing doesn't exist on python 
        self._op = _op                              # If you put two: "__hola" python rename it to _Value__hola, so if you write __hola throw an error saying it doesn't exist (but you can access it with the weird name _Value__hola )
        self._backward = lambda : None      #We define a function that does nothing
        self._label = label
        self.grad = 0.0
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)     # With this we can introduce just scalars as other and everything should work 
        out = Value(self.data + other.data, (self,other), '+')
        
        def _backward(): 
            self.grad += out.grad*1     #Chain rule with add / We use += because the gradient is added if it is use this node more than one: a+a=b => grad(a) = grad(b) * 2 = grad(b) + grad(b)
            other.grad += out.grad*1
        out._backward = _backward
        return out
    
    def __mul__ (self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward(): 
            self.grad += out.grad * other.data     #Chain rule with mul 
            other.grad += out.grad* self.data
        out._backward = _backward

        return out

    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out
    
    def __sub__(self, other): # self - other
        return self + (-other)
    
    def __neg__(self): # -self
        return self * -1
    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __radd__(self, other): # other + self
        return self + other
    


    def tanh(self): 
        x = self.data
        t = ((math.exp(2*x)-1) / (math.exp(2*x)+1))      #The mathematical expression of tanh
        out = Value(t, (self,), 'tanh')

        # The derivate of tanh
        def _backward(): 
            self.grad += (1 - t**2)* out.grad   # 1- tanh(x)^2  * parent 
        out._backward = _backward

        return out 
    
    # I also impliment relu. Relu its the standard for nn
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')      # zero if its less than zero, else is the identity

        def _backward(): 
            self.grad += (0 if out.data<=0 else 1)* out.grad   # 1- tanh(x)^2  * parent 
        out._backward = _backward
        return out


    # Use topological sort to create the order to call the backward 
    def backward(self):
        topo = []
        visited = set()
        def buildTopo(v):
            if v not in visited: 
                visited.add(v)
                for childrens in v._prev:
                    buildTopo(childrens)
                topo.append(v)
        buildTopo(self)     #creamos la lista que vamos a recorrer
        self.grad = 1.0    #L siempre tiene derivada 1 con respecto de L 
        for node in reversed(topo):
            node._backward()


        

