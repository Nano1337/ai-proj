import random
from backprop import Value
import pandas as pd
import numpy as np

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

def MSE(y_pred, y_true):
    """
    Calculate the mean squared error between the predicted and true labels.
    
    :param y_pred: Predicted values, a numpy array of shape (n_samples,)
    :param y_true: True values, a numpy array of shape (n_samples,)
    :return: The mean squared error.
    """
    mse = np.mean((y_pred - y_true) ** 2)
    return mse

# train for one epoch
def train_epoch(model, batch_size=None):
    
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))

    # Calculate Mean Squared Error (MSE) Loss
    losses = [((yi - scorei)**2) for yi, scorei in zip(yb, scores)]
    total_loss = sum(losses) * (1.0 / len(losses))
    
    return total_loss

if __name__ == "__main__":

    # make ann
    input_dim = 10
    hidden_dim = [8]
    output_dim = 1
    model = MLP(input_dim, hidden_dim, output_dim)

    # read and extract data, labels
    df = pd.read_csv('data.txt', sep='\t', encoding='utf-16')
    X = df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']].to_numpy()
    y = df['utility'].to_numpy() 

    # normalize design matrix
    X = (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True)
   
    # run training
    epochs = 100  # Number of training epochs
    learning_rate = 0.05  # Learning rate for weight updates
    batch_size = 250

    for k in range(100):
        
        # forward
        total_loss = train_epoch(model, batch_size=250)
        
        # backward
        model.zero_grad()
        total_loss.backwards()
        
        # update with stochastic gradient descent
        # learning_rate = 1.0 - 0.1*k/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 1 == 0:
            print(f"step {k} loss {total_loss.data}")
        