import torch
import torch.nn as nn
class NeRF_ViewDependent(nn.Module):
    def __init__(self,embedding_dim_pos=39,embedding_dim_dir=27):
        super().__init__()
        self.linear_tl1=nn.Linear(embedding_dim_pos,128)
        self.linear_tl2=nn.Linear(128,128)
        self.linear_tl3=nn.Linear(128,128)
        
        self.linear_al=nn.Linear(128,1)
        
        self.linear_bl1=nn.Linear(128+embedding_dim_dir,64)
        self.linear_bl2=nn.Linear(64,3)
        
        self.relu=nn.ReLU()
    def forward(self,x,d):
        out_1=self.relu(self.linear_tl3(self.relu(self.linear_tl2(self.relu(self.linear_tl1(x))))))
        
        sigma=self.linear_al(out_1)
        
        concatenated_out1=torch.cat((out_1,d),dim=-1)
        
        rgb=self.linear_bl2(self.relu(self.linear_bl1(concatenated_out1)))
        
        return torch.cat([rgb,sigma],dim=-1)
