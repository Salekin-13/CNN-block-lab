import torch
import torch.nn as nn

class LeNet(nn.Module):
    """
    The size of the output of the convolution layer: [(nh-kh+ph+sh)/sh , (nw-kw+pw+sw)/sw]
    where nh, nw -> input height and weight
    kh, kw -> kernel
    ph, pw -> total padding (rows or column)/2
    sh, sw -> stride
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # 3x32x32 -> 6x28x28 ; bordering pixel values are less relevant
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 6x28x28 -> 6x14x14
                                                          #usually stride is matched with kernel size in pooling
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5) # 6x14x14 -> 16x10x10
        self.fc1 = nn.Linear(in_features=400, out_features=120) # 16*5*5 -> 120
        self.fc2 = nn.Linear(in_features=120, out_features=84) 
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes) #number of classes


    def forward(self, x):
        """ defining the forward pass """
        x = torch.tanh(self.conv1(x)) #classical lenet-5 architecture used a version of sigmoid activation, though contemporarily, relu has better result in terms of gradient flow and generalization
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5) #flatten x into a 2d tensor (batch size, 400) for the fully connected layer 
                                # batch size is automatically inferred using -1 to keep the number of elements same
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)

        return x