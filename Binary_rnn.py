import numpy as np
import torch
from torch.nn import Module, RNN, Linear
from torch.nn.functional import linear, conv2d, hardtanh
import labeling
import torch.nn as nn
#from torch.autograd import Variable
import torch.autograd as autograd
import sys
from matplotlib import pyplot as plt

sequence_length = 1
input_size = 120
hidden_size = 240
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

    def __init__(self, num_classes=1):
        super(PacketRnn, self).__init__()

        self.features = nn.Sequential(
            B_RNN(input_size,hidden_size),
            RNNLinear(hidden_size, num_classes),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        for m in self.modules():
            if isinstance(m, RNN):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.h_t = torch.autograd.Variable(torch.zeros(1, hidden_size))
                nn.init.kaiming_normal_(self.weight_ih_org, mode='fan_out')
                nn.init.kaiming_normal_(self.weight_hh_org, mode='fan_out')
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

# class RNN_cell(nn.module):
#
#     def __init__(self,input_size,hidden_size, n_layers = 1):
#
#         super(RNN_cell, self).__init__()
#         self.input_size  = input_size
#         self.hidden_size = hidden_size
#         self.n_layers = 1
#         self.x2h_i = torch.nn.Linear(input_size, hidden_size)
#         self.h2o   = torch.nn.Linear(hidden_size, hidden_size)
#         # self.register_buffer('ih_weight_org', self.weight_ih_l0.data.clone())
#         # self.register_buffer('hh_weight_org', self.weight_hh_l0.data.clone())
#         self.register_buffer('ih_weight_org', self.x2h_i.weight.data.clone())
#         self.register_buffer('hh_weight_org', self.h2o.weight.data.clone())
#     def forward(self, input):
#         input.data = input_Binarize(self.input)
#         self.x2h_i.weight.data = Binarize(self.ih_weight_org)
#         self.h2o.weight.data = Binarize(self.hh_weight_org)
#         #todo hidden value binarize
#         middle = self.x2h_i(self.input) + self.h2o(self.last_hidden)
#         output = StraightThroughEstimator(middle)
#         # output = nn.sign(middle)
#         h_t = output.clone()
#
#         return output, h_t
#
#     def initHidden(self):
#         return torch.autograd.Variable(torch.zeros(1, self.hidden_size))
#
#     def weights_init(self,model):
#
#         classname = model.__class__.__name__
#         if classname.find('Linear') != -1:
#             model.weight.data.normal_(0.0, 0.02)
#             model.bias.data.fill_(0)

class RNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(RNNLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = linear(input, self.weight)

        return out

# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0).mul_(2).sub_(1).float()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return hardtanh(grad_output)
#
# class StraightThroughEstimator(nn.Module):
#     def __init__(self):
#         super(StraightThroughEstimator, self).__init__()
#
#     def forward(self, x):
#         x = STEFunction.apply(x)
#         return x

#todo weight initialization

class B_RNN(RNN) :
    def __init__(self, *kargs, **kwargs):
        super(B_RNN, self).__init__(*kargs, **kwargs)
        self.input_size  = 120
        self.hidden_size = 240
        self.batch_size = 10
        self.n_layers = 1
    #weight initialize
        self.h_t = None
        self.weight_ih = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
        self.weight_hh = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
        self.register_buffer('weight_ih_org', self.weight_ih.data.clone())
        self.register_buffer('weight_hh_org', self.weight_ih.data.clone())

    def forward(self, input):
        # for self.h_t == None :
            # self.h_t = torch.autograd.Variable(torch.zeros(1, self.hidden_size))
        data_ih = torch.zeros(1, hidden_size)
        data_hh = torch.zeros(1, hidden_size)
        middle = torch.zeros(1, hidden_size)

        # 1과 0이던 input을 1과 -1인 형식으로 변경
        input = input_Binarize(input)
        self.h_t = Binarize(self.h_t)

        self.x2h_i.weight.data = Binarize(self.weight_ih_org)
        self.h2o.weight.data = Binarize(self.weight_hh_org)

        for i in range(self.batch_size):

            data_ih = torch.dot(input[i], self.weight_ih)
            data_hh = torch.dot(self.h_t, self.weight_hh)
            middle = data_ih + data_hh
            #ste function?
            self.h_t = torch.sign(middle)
        # output = nn.sign(middle)

        return self.h_t

class B_RNNtrainer():
    def __init__(self, model, bit= 120, lr=0.01, device=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.bit = bit
        self.device = device

    def train_step(self):
        #data = torch.zeros(182000, self.bit)
        data = torch.zeros(10000, self.bit)
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
            if t ==1000 :
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
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    model = B_RNN(input_size= 120 ,hidden_size = 240, batch_size = 10)
    model.init_w()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # sample input

    B_RNN = B_RNNtrainer(model, bit = 120, device='cuda')
    #optimizer = torch.optim.Adam(Packetbnn.parameters(), lr=0.002, weight_decay=1e-7)
    epoch_losses= B_RNN.train_step()
    # b = []
    # c = []
    # for i in epoch_losses:
    #     k = 0
    #     if k%100 == 0 :
    #         b.append(i[0])
    #         c.append(i[1])
    #     k +=1

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(b, c, 'k', 9, label='bnn_loss sum')
    # plt.show()
    # final_weight = Packetbnn.features[0].weight
    # final_weight_org = Packetbnn.features[0].weight_org
    #
    # print(ini_weight[0])
    # print(final_weight[0])
    # print(ini_weight[0]-final_weight[0])
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
