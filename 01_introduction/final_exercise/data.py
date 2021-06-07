import torch

def mnist():
    # exchange with the real mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 785) 
    return train, test
