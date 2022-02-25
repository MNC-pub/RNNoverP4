import numpy as np
import torch
from torch.nn import Module, RNN, Linear, LSTM
from torch.nn.functional import linear, conv2d, hardtanh
import labeling
import torch.nn as nn
#from torch.autograd import Variable
import torch.autograd as autograd
import sys
from matplotlib import pyplot as plt


__all__ = ['packetRnn']

sequence_length = 1
input_size = 120
hidden_size = 128
num_layers = 1
num_classes = 2
batch_size = 10
learning_rate = 0.01

def Binarize(tensor):
    #result = (tensor-0.5).sign()
    return tensor.sign()

def input_Binarize(tensor):
    return tensor.sub_(0.5).sign()

class PacketRnn(nn.Module):
    input_size = 120
    hidden_size = 128

    def __init__(self, num_classes=1):
        super(PacketRnn, self).__init__()

        self.features = nn.Sequential(
            RNN_cell(input_size,hidden_size),
            RNN_cell(input_size, hidden_size),
            RNN_cell(input_size, hidden_size),
            RNN_cell(input_size, hidden_size),
            RNN_cell(input_size, hidden_size),
            RNNLinear(hidden_size, num_classes),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(self.h_0, mode='fan_out')
                nn.init.kaiming_normal_(self.weight_org, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.GroupNorm):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a= 1., b= 1.)
                nn.init.zeros_(m.bias)
        return

class RNN_cell(nn.module):

    def __init__(self,input_size,hidden_size, n_layers = 1):

        super(RNN_cell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1
        self.x2h_i = torch.nn.Linear(input_size, hidden_size)
        self.h2o   = torch.nn.Linear(hidden_size, hidden_size)
        self.register_buffer('weight_org', self.weight_ih_l0.data.clone())
        self.register_buffer('weight_org', self.weight_hh_l0.data.clone())

    def forward(self, input):
        input.data = input_Binarize(self.input)
        weight_ih_l0 = Binarize(self.weight_ih_l0)
        weight_hh_l0 = Binarize(self.weight_hh_l0)
        #todo hidden value binarize
        middle = self.x2h_i(self.input) + self.h2o(self.last_hidden)
        output = StraightThroughEstimator(middle)
        # output = nn.sign(middle)
        h_t = output.clone()

        return output, h_t

    def initHidden(self):
        return torch.autograd.Variable(torch.zeros(1, self.hidden_size))

    def weights_init(self,model):

        classname = model.__class__.__name__
        if classname.find('Linear') != -1:
            model.weight.data.normal_(0.0, 0.02)
            model.bias.data.fill_(0)

# class Binary_AF:
#     def __init__(self, x):
#         self.x = x
#
#     def forward(self):
#         self.x[self.x <= 0] = 0
#         self.x[self.x > 0] = 1
#         return self.x
#
#     def backward(self):
#         return self.x

class RNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(RNNLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = linear(input, self.weight)

        return out

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).mul_(2).sub_(1).float()

    @staticmethod
    def backward(ctx, grad_output):
        return hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

#todo weight initialization

# class B_RNN(RNN):
#
#     def __init__(self, *kargs, **kwargs):
#         super(B_RNN, self).__init__(*kargs, **kwargs)
#         self.register_buffer('weight_org', self.weight.data.clone())
#
#     def forward(self, input):
#
#         input.data = Binarize(input.data)
#
#         self.weight.data = Binarize(self.weight_org)
#
#         out = RNN(input, self.h_0)
#         #
#         # if not self.bias is None:
#         #     self.bias.org=self.bias.data.clone()
#         #     out += self.bias.view(1, -1, 1, 1).expand_as(out)
#
#         return out


class Bnntrainer():
    def __init__(self, model, bit, lr=0.01, device=None):
        super().__init__()
        self.model = model
        self.bit = bit
        self.lr = lr
        self.device = device

    def train_step(self, optimizer):
        #data = torch.zeros(182000, self.bit)
        data = torch.zeros(100000, self.bit)
        epoch_losses = []
        epoch_loss = 0
        input = torch.zeros(1,1,1,self.bit)
        label = labeling.label()
        f = open("output.txt", "r")
        content = f.readlines()
        #t means packet sequence
        data_target = [[]]
        t = 0
        for line in content:
            k = 0
            if t ==100000 :
                break
            for i in line:
                if i.isdigit() == True:
                    data[t][k] = int(i)
                    k += 1

            input[0][0] = data[t]
            target = torch.tensor(label[t])
            #input, target = input.to(self.device), target.to(self.device)
            output = self.model(input)

            loss = (output-target).pow(2).sum()
            loss = autograd.Variable(loss, requires_grad=True)
            #losses.append(loss.item())
            epoch_loss +=loss.item()
            epoch_losses.append([t,epoch_loss])
            optimizer.zero_grad()
            loss.backward()

            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight.data.copy_(p.weight_org)
            optimizer.step()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    #p.weight_org.data.copy_(p.weight.data.clamp_(-1, 2))
                    p.weight_org.copy_(p.weight.data.clamp_(-1, 1))
            t +=1
        return epoch_losses

if __name__ == '__main__':
    #data load
    # f = open("output.txt", "r")
    # content = f.readlines()
    torch.set_printoptions(threshold=50000)
    torch.set_printoptions(linewidth=20000)
    # cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if cuda else 'cpu')

    model = PacketRnn(input_size, hidden_size, num_layers, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    device = torch.device('cpu')
    # print(device)
    bit = 120
    Packetbnn = packetbnn()
    # model = eval("packetbnn")()
    # model.to(device)
    Packetbnn.to(device)
    Packetbnn.init_w()

    # sample input

    Bnn = Bnntrainer(Packetbnn, bit=120, device='cuda')
    optimizer = torch.optim.Adam(Packetbnn.parameters(), lr=0.002, weight_decay=1e-7)
    ini_weight = Packetbnn.features[0].weight.clone()
    ini_weight_org = Packetbnn.features[0].weight_org.clone()
    epoch_losses= Bnn.train_step(optimizer)
    b = []
    c = []
    for i in epoch_losses:
        k = 0
        if k%100 == 0 :
            b.append(i[0])
            c.append(i[1])
        k +=1

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(b, c, 'k', 9, label='bnn_loss sum')
    # plt.show()
    final_weight = Packetbnn.features[0].weight
    final_weight_org = Packetbnn.features[0].weight_org

    print(ini_weight[0])
    print(final_weight[0])
    print(ini_weight[0]-final_weight[0])
    #
    # print(ini_weight_org[0])
    # print(final_weight_org[0])
    # print(ini_weight_org[0]-final_weight_org[0])

    sys.stdout = open('weight.txt', 'w')

    print(Packetbnn.features[0].weight)

    # print(Binarize(Packetbnn.features[0].weight).byte())
    # print(Binarize(Packetbnn.features[4].weight).byte())
    print(Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int())
    print(Binarize(Packetbnn.features[4].weight).add_(1).div_(2).int())
