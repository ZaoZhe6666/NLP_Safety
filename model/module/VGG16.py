import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 32 * 32
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 32 * 32

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn11 = nn.BatchNorm2d(64)

        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 32* 32

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn12 = nn.BatchNorm2d(64)

        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 16 * 16

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 16 * 16

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn21 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 16 * 16

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn22 = nn.BatchNorm2d(128)

        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 8 * 8

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 8 * 8

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn31 = nn.BatchNorm2d(256)

        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 8 * 8

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn32 = nn.BatchNorm2d(256)

        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 8 * 8

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn33 = nn.BatchNorm2d(256)

        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 4 * 4

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 4 * 4

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn41 = nn.BatchNorm2d(512)

        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 4 * 4

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn42 = nn.BatchNorm2d(512)

        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 4 * 4

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn43 = nn.BatchNorm2d(512)

        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 2 * 2

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 2 * 2

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn51 = nn.BatchNorm2d(512)

        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 2 * 2

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn52 = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 2 * 2

        # multiple BN to investigate data distribution of adversarial examples with different Lp perturbations
        self.bn53 = nn.BatchNorm2d(512)

        self.maxpool5 = nn.MaxPool2d((2, 2))  # pooling 512 * 1 * 1

        # view

        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)
        # softmax 1 * 1 * 1000
      
        
    def forward(self, input):
        in_size = input.size(0)
        
        out = self.conv1_1(input)  # 222
        
        # multiple BN structures
        out = self.bn11(out)
        
        out = F.relu(out)
        
        out = self.conv1_2(out)  # 222
        
        # multiple BN structures
        out = self.bn12(out)
        
        out = F.relu(out)
        
        out = self.maxpool1(out)  # 112
        
        out = self.conv2_1(out)  # 110

        # multiple BN structures
        out = self.bn21(out)

        out = F.relu(out)
        out = self.conv2_2(out)  # 110

        # multiple BN structures
        out = self.bn22(out)

        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54

        # multiple BN structures
        out = self.bn31(out)

        out = F.relu(out)
        out = self.conv3_2(out)  # 54

        # multiple BN structures
        out = self.bn32(out)

        out = F.relu(out)
        out = self.conv3_3(out)  # 54

        # multiple BN structures
        out = self.bn33(out)

        out = F.relu(out)
        out = self.maxpool3(out)  # 28
        
        out = self.conv4_1(out)  # 26

        # multiple BN structures
        out = self.bn41(out)

        out = F.relu(out)
        out = self.conv4_2(out)  # 26

        # multiple BN structures
        out = self.bn42(out)

        out = F.relu(out)
        out = self.conv4_3(out)  # 26

        # multiple BN structures
        out = self.bn43(out)

        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12

        # multiple BN structures
        out = self.bn51(out)

        out = F.relu(out)
        out = self.conv5_2(out)  # 12

        # multiple BN structures
        out = self.bn52(out)

        out = F.relu(out)
        out = self.conv5_3(out)  # 12

        # multiple BN structures
        out = self.bn53(out)

        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        #out = F.log_softmax(out, dim=1)

        return out