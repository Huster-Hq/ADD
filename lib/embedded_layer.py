import torch.nn as nn


class embed_layer(nn.Module):
    def __init__(self):
        super(embed_layer, self).__init__()
        down_channel=128
        self.conv_embedd= nn.Conv2d(in_channels=2048, out_channels=down_channel, kernel_size=1,stride=1, bias=False,)#kernel_size=3,stride=2
    
    def forward(self, x):
        x=self.conv_embedd(x)
        return x.flatten(2)
    