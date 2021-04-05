import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def fit(loss, params, X, Y, Xval, Yval, opt, opt_kwargs={"lr":1e-3}, batch_size=128, epochs=100, verbose=False, callback=None):
    """

    Arguments:
        loss: given x and y in batched form, evaluates loss.
        params: list of parameters to optimize.
        X: input data, torch tensor.
        Y: output data, torch tensor.
        Xval: input validation data, torch tensor.
        Yval: output validation data, torch tensor.
    """

    train_dset = TensorDataset(X, Y)
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    lr = opt_kwargs.pop('lr')
    opt = opt(params, lr=lr, **opt_kwargs)
    
    train_losses = []
    val_losses = []
    
    param_updates = torch.zeros((epochs+1,len(params[0])))
    param_updates[0] = params[0]
    
    for epoch in range(epochs):
        if (epoch + 1) % int(epochs/2) == 0:
            lr = lr / 2
            opt = torch.optim.Adam(params, lr=lr, **opt_kwargs)
            print('new lr:', lr)
        with torch.no_grad():
            val_losses.append(loss(Xval, Yval).item())
        if verbose:
            print("%03d | %3.5f" % (epoch + 1, val_losses[-1]))
        batch = 1
        train_losses.append([])
        for Xbatch, Ybatch in train_loader:
            opt.zero_grad()
            l = loss(Xbatch, Ybatch)
            l.backward()
            opt.step()
            train_losses[-1].append(l.item())
            if verbose:
                print("batch %03d / %03d | %3.5f" %
                      (batch, len(train_loader), np.mean(train_losses[-1])))
            batch += 1
            if callback is not None:
                callback()
                
        param_updates[epoch+1,:] = params[0]
        
    return val_losses, train_losses, param_updates