import torch.nn as nn
import torch.nn.functional as F

import sys

class BasicRes(nn.Module):
    def __init__(self):
        super(BasicRes, self).__init__()
        self.block1 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(1, 1, 7)
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),  # D_in = (W_in, H_in, C_in) = D_out = (W_out, H_out, C_out)
            nn.Conv2d(1, 1, 3, padding=1),  # D_in = D_out
            nn.Conv2d(1, 2, 3, stride=2, padding=1),    # D_out = (W_in/2, H_in/2, 2)
            nn.Conv2d(2, 2, 3, padding=1),  # D_in = (W_in, H_in, 2) = D_out
            nn.Conv2d(2, 2, 3, padding=1),  # D_in = D_out
            nn.Conv2d(2, 4, 3, stride=2, padding=1),    # D_out = (W_in/2, H_in/2, 4)
            nn.Conv2d(4, 4, 3, padding=1),  # D_in = D_out
            nn.Conv2d(4, 4, 3, padding=1),  # D_in = D_out
            nn.Conv2d(4, 8, 3, stride=2, padding=1),    # D_out = (W_in/2, H_in/2, 8) 
            nn.Conv2d(8, 8, 3, padding=1),  # D_in = D_out
            nn.Conv2d(8, 8, 3, padding=1),  # D_in = D_out
            nn.Conv2d(8, 16, 3, stride=2, padding=1)    # D_out = (W_in/2, H_in/2, 12) 
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.block1(x))

        for layer in self.res_block:
            residual = x
            x = F.relu(layer(x))
            x += residual
        
        # DEBUG
        # Print out the size of the final input to the fully connected layer,
        # save it, and modify the code
        print(x.size())
        sys.exit()
