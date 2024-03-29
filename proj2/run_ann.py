"""
Team Members (Group 3):
- Haoli Yin
- Lydia Liu
- Richard Song
- Shreya Reddy

How to Run Code:
Create a python virtual environment
Install dependencies (Linux instructions, slightly different for Windows)
Replace hidden_dim (marked w/ TODO) w/ 4 or 8
Run code
```bash 
"""
import random
from backprop import Value
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils_ann import kfold_indices, normalize

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

# train for one epoch
def run_epoch(model, X, y, batch_size=None):
    
    # make batches
    if batch_size is None: # use all data
        Xb, yb = X, y
    else: # use batch size
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]
    
    # forward the model to get scores
    scores = list(map(model, inputs))
    
    if batch_size is None: # for validation
        mres = [(abs(yi - scorei.data)/yi) for yi, scorei in zip(yb, scores)]
        return np.mean(np.array(mres))

    else: # for training
        # Calculate Mean Squared Error (MSE) Loss
        losses = [((yi - scorei)**2) for yi, scorei in zip(yb, scores)]
        total_loss = sum(losses) * (1.0 / len(losses))
        return total_loss

def train_model(model, X_train, y_train): 
    # run training
    epochs = 75  # Number of training epochs
    learning_rate = 0.1  # Learning rate for weight updates
    batch_size = 250

    for k in tqdm(range(epochs)):
        
        # forward
        total_loss = run_epoch(model, X_train, y, batch_size=250)
        
        # backward
        model.zero_grad()
        total_loss.backwards()
        
        # update with stochastic gradient descent
        for p in model.parameters():
            p.data -= learning_rate * p.grad

    return model, total_loss.data

if __name__ == "__main__":

    # ann hyperparameters
    input_dim = 10
    hidden_dim = [8] #TODO: Replace w/ 4 or 8
    output_dim = 1

    # read and extract data, labels
    df = pd.read_csv('orig_data.txt', sep='\t', encoding='utf-8')
    X = df[['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9']].to_numpy()
    y = df['utility'].to_numpy() 

    # 5-fold cross-validation
    k = 5
    fold_indices = kfold_indices(X, k)

    fold_losses = []
    # NOTE: this takes about 2 minutes to completely run
    count = 1
    for train_indices, test_indices in fold_indices:
        # reset model and create train/val split
        model = MLP(input_dim, hidden_dim, output_dim)
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        X_train, X_test = normalize(X_train, X_test)

        # train model
        model, loss = train_model(model, X_train, y_train)

        # val model
        loss = run_epoch(model, X_test, y_test)
        print(f'Fold {count} MRE Loss: {loss}')
        fold_losses.append(loss)
        count += 1

    fold_losses = np.array(fold_losses)
    print('The 5 fold MRE losses are:', fold_losses)
    print(f'Mean of MRE losses: {np.mean(fold_losses)}')
    print(f'Variance of MRE losses: {np.var(fold_losses)}')





        


