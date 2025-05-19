import select
from turtle import forward
from matplotlib.pyplot import cla
import torch
import torch.nn as nn

### A base CNN model
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


### modified the CNN with increased layers,batchnorm, relu activation and dilation
class mod_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(mod_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2) ## 3x32x32 -> 6x32x32
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 6x32x32 -> 6x16x16

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=2) # 6x16x16 -> 16x18x18
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, padding=2) # 16x9x9 -> 16x9x9
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2, dilation=2) # 16x9x9 -> 32x5x5
        self.bn4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120) # 32*5*5 -> 120
        self.fc2 = nn.Linear(in_features=120, out_features=84) 
        self.drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes) #number of classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))

        #x = self.pool(x)
        #print(x.shape)

        x = x.view(-1, 32*5*5)
        x = torch.relu(self.fc1(x))
        x = self.drop(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

### using depthwise separable convolution for a more lightweight model

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthConv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding, dilation=dilation)
        self.pointConv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthConv(x)
        out = self.pointConv(out)
        return out


class mod_CNN_depthwise(nn.Module):
    def __init__(self, num_classes=10):
        super(mod_CNN_depthwise, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels=3, out_channels=6, kernel_size=5, padding=2) ## 3x32x32 -> 6x32x32
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 6x32x32 -> 6x16x16

        self.conv2 = DepthwiseSeparableConv(in_channels=6, out_channels=16, kernel_size=3, padding=2) # 6x16x16 -> 16x18x18
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = DepthwiseSeparableConv(in_channels=16, out_channels=16, kernel_size=5, padding=2) # 16x9x9 -> 16x9x9
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = DepthwiseSeparableConv(in_channels=16, out_channels=32, kernel_size=5, padding=2, dilation=2) # 16x9x9 -> 32x5x5
        self.bn4 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=32*5*5, out_features=120) # 32*5*5 -> 120
        self.fc2 = nn.Linear(in_features=120, out_features=84) 
        self.drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes) #number of classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))

        x = x.view(-1, 32*5*5)
        x = torch.relu(self.fc1(x))
        x = self.drop(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


### using residual blocks to improve model performance

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity

        return self.relu(out)
    
class mod_CNN_depth_res(nn.Module):
    def __init__(self, num_classes=10):
        super(mod_CNN_depth_res, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels=3, out_channels=6, kernel_size=5, padding=2) ## 3x32x32 -> 6x32x32
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # 6x32x32 -> 6x16x16

        self.conv2 = DepthwiseSeparableConv(in_channels=6, out_channels=16, kernel_size=3, padding=1) # 6x16x16 -> 16x16x16
        self.bn2 = nn.BatchNorm2d(16)

        self.res1 = ResidualBlock(16)

        self.conv3 = DepthwiseSeparableConv(in_channels=16, out_channels=32, kernel_size=5, padding=2) # 16x8x8 -> 32x8x8
        self.bn3 = nn.BatchNorm2d(32)

        self.res2 = ResidualBlock(32)

        self.conv4 = DepthwiseSeparableConv(in_channels=32, out_channels=64, kernel_size=5, padding=2, dilation=2) # 32x8x8 -> 64x4x4
        self.bn4 = nn.BatchNorm2d(64)

        self.res3 = ResidualBlock(64)

        self.fc1 = nn.Linear(in_features=64*4*4, out_features=120) # 64*4*4 -> 120
        self.fc2 = nn.Linear(in_features=120, out_features=84) 
        self.drop = nn.Dropout(0.3)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes) #number of classes

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.res1(x)

        x = torch.relu(self.bn3(self.conv3(x)))

        x = self.res2(x)

        x = torch.relu(self.bn4(self.conv4(x)))

        x = self.res3(x)

        x = x.view(-1, 64*4*4)
        x = torch.relu(self.fc1(x))
        x = self.drop(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
