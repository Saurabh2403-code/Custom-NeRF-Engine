import torch
import torch.nn as nn

class networkv1(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear_stack=nn.Sequential(
        nn.Linear(2,50),
        nn.ReLU(),
        nn.Linear(50,50),
        nn.ReLU(),
        nn.Linear(50,50),
        nn.ReLU(),
        nn.Linear(50,3)
    )
  def forward(self,x):
    return self.linear_stack(x)
  

class networkv2(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.linear_stack=nn.Sequential(
            nn.Linear(42,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,3),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.linear_stack(x)