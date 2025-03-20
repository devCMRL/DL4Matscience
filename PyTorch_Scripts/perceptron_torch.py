import torch #type: ignore
import torch.nn as nn #type: ignore

class PerceptronTorch(nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super(PerceptronTorch, self).__init__()
        # Linear transformation
        self.linear = nn.Linear(input_dim, output_dim)
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, x):
        out = self.linear(x)
        return self.activation(out)
