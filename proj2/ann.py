import random
from backprop import Value
import pandas as pd

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.layers = []
        # input 
        self.layers.append(Layer(input_dim, hidden_dims[0], nonlin=True))
        # hidden layers
        for i in range(1, len(hidden_dims)):
            self.layers.append(Layer(hidden_dims[i-1], hidden_dims[i], nonlin=True))
        # output
        self.layers.append(Layer(hidden_dims[-1], output_dim, nonlin=False))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

if __name__ == "__main__":
    input_dim = 10
    hidden_dim = [4]
    output_dim = 1
    model = MLP(input_dim, hidden_dim, output_dim)

    df = pd.read_csv('data.txt', sep='\t', encoding='utf-16')
    
    X = df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']].to_numpy()
    labels = df['utility'].to_numpy() 
    print(X.shape, labels.shape)
