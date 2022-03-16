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
input_size = 128
hidden_size = 128
# num_layers = 1
num_classes = 2
batch_size = 1
learning_rate = 0.001

def Binarize(tensor):
    #result = (tensor-0.5).sign()
    return tensor.sign()

def input_Binarize(tensor):
    return tensor.sub_(0.5).sign()

class PacketRnn(nn.Module):

    def __init__(self, num_classes=2):
        super(PacketRnn, self).__init__()

        self.features = nn.Sequential(
            B_RNN(input_size,hidden_size),
            RNNLinear(hidden_size, 1),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        for m in self.modules():
            if isinstance(m, RNN):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #m.h_t = torch.autograd.Variable(torch.zeros(10, batch_size, hidden_size))
                nn.init.kaiming_normal_(m.weight_ih_l0, mode='fan_out')
                nn.init.kaiming_normal_(m.weight_hh_l0, mode='fan_out')
                nn.init.kaiming_normal_(m.weight_ih_l0_org, mode='fan_out')
                nn.init.kaiming_normal_(m.weight_hh_l0_org, mode='fan_out')
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)
            # elif isinstance(m, nn.GroupNorm):
            #     nn.init.ones_(m.weight)
            #     nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a= .9, b= 1.)
                nn.init.zeros_(m.bias)
        return

class RNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(RNNLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = linear(input, self.weight)
        out = out/10
        return out

#todo weight initialization

class B_RNN(RNN) :
    def __init__(self, *kargs, **kwargs):
        super(B_RNN, self).__init__(*kargs, **kwargs)
        #self.batch_size = 1
        self.n_layers = 1
        self.sequence_length = sequence_length
    #weight initialize
        self.weight_ih = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
        self.weight_hh = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
        self.register_buffer('weight_ih_l0_org', self.weight_ih_l0.data.clone())
        self.register_buffer('weight_hh_l0_org', self.weight_hh_l0.data.clone())

    def forward(self, input):
        # for self.h_t == None :
            # self.h_t = torch.autograd.Variable(torch.zeros(1, self.hidden_size))
        data_ih = torch.zeros(1, hidden_size)
        data_hh = torch.zeros(1, hidden_size)
        middle = torch.zeros(1, hidden_size)
        self.h_t = torch.autograd.Variable(torch.zeros(11, batch_size, hidden_size))

        # 1과 0이던 input을 1과 -1인 형식으로 변경
        input = input_Binarize(input)
        self.h_t = Binarize(self.h_t)

        self.weight_ih_l0.data = Binarize(self.weight_ih_l0_org)
        self.weight_hh_l0.data = Binarize(self.weight_hh_l0_org)
        #print("ih",self.weight_ih_l0.size())
        #print("hh",self.weight_hh_l0.size())
        for i in range(0,10):
            data_ih = torch.matmul(input[0][i], self.weight_ih_l0)
            data_hh = torch.matmul(self.h_t[i], self.weight_hh_l0)
            middle = data_ih + data_hh
            #ste function?
            self.h_t[i+1] = torch.sign(middle)
        # output = nn.sign(middle)

        return self.h_t[10]

class B_RNNtrainer():
    def __init__(self, model, bit= 128, lr=0.001, device=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.bit = bit
        self.device = device

    def train_step(self,optimizer):
        #data = torch.zeros(182000, self.bit)
        data = torch.zeros(100000, self.bit)
        target = torch.zeros(180000, 1)
        Target=0
        epoch_losses = []
        epoch_loss = 0
        f = open("11label.txt", "r")
        content = f.readlines()
        label_Index = 0
        for line in content:
            for i in line:
                if i.isdigit() == True:

                    # if i == 0 :
                    #     target[label_Index][0] = torch.tensor(int(i))
                    # elif i == 1:
                    #     target[label_Index][1] = torch.tensor(int(i))
                    target[label_Index] = torch.tensor(int(i))
                    #target[label_Index] = torch.tensor(int(i))
                    label_Index +=1
        input = torch.zeros(1,10,self.bit)
        f = open("final.txt", "r")
        content = f.readlines()
        #t means packet sequence
        t = 0
        for line in content:
            k = 0
            if t ==10000 :
                break
            for i in line:
                if i.isdigit() == True:
                    data[t][k] = int(i)
                    k += 1
            if t < 9:
                input[0][t] = data[t]
            elif t >= 9 :
                #input[0][t] = data[t]
                #input, target = input.to(self.device), target.to(self.device)

                if t > 9 :
                    for index in range(9):
                        input[0][index] = input[0][index+1]
                input[0][9] = data[t]

                output = self.model(input)
                # print("output",output)
                # print("target",target[t])
                loss = (output-target[t]).pow(2).sum()

                #Target = target[int((t-9)/10)].long
                #loss = criterion(output, Target)
                loss = autograd.Variable(loss, requires_grad=True)
                #losses.append(loss.item())
                epoch_loss +=loss.item()
                epoch_losses.append([t,epoch_loss])
                optimizer.zero_grad()
                loss.backward()

                for p in self.model.modules():
                    if hasattr(p, 'weight_org'):
                        p.weight_ih_l0.data.copy_(p.weight_ih_l0_org)
                        p.weight_hh_l0.data.copy_(p.weight_hh_l0_org)
                optimizer.step()
                for p in self.model.modules():
                    if hasattr(p, 'weight_org'):
                        p.weight_ih_l0_org.copy_(p.weight_ih_l0.data.clamp_(-1, 1))
                        p.weight_hh_l0_org.copy_(p.weight_hh_l0.data.clamp_(-1, 1))
            t +=1
        return epoch_losses

class test_RNNtrainer():
    def __init__(self, model, bit= 128, lr=0.001, device=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.bit = bit
        self.device = device

    def test(self, final_w, test_start) :

        self.model.features[0].weight_ih_l0 = torch.nn.Parameter(final_w)

        accuracy = 0
        data = torch.zeros(100000, 128)
        target = torch.zeros(180000, 1)

        f = open("11label.txt", "r")
        content = f.readlines()
        label_Index = 0
        for line in content:
            for i in line:
                if i.isdigit() == True:
                    # if i == 0 :
                    #     target[label_Index][0] = torch.tensor(int(i))
                    # elif i == 1:
                    #     target[label_Index][1] = torch.tensor(int(i))
                    target[label_Index] = torch.tensor(int(i))
                    # target[label_Index] = torch.tensor(int(i))
                    label_Index += 1
        input = torch.zeros(1, 10, 128)
        f = open("final.txt", "r")
        content = f.readlines()
        # t means packet sequence
        t = 0
        for line in content:
            k = 0

            if t == test_start+10000:
                break
            for i in line:
                if i.isdigit() == True:
                    data[t][k] = int(i)
                    k += 1
            if t >= test_start  :
                if t < test_start+9 :
                    input[0][t-test_start] = data[t]
                else :

                    if t > test_start+9:
                        for index in range(9):
                            input[0][index] = input[0][index + 1]
                    input[0][9] = data[t]
                    output = self.model(input)

                    Binary_output = output.sign().div_(2).add_(0.5)
                    print("Binary_output", output)
                    print("target[t]", target[t])
                    if int(Binary_output) == target[t] :

                        accuracy +=0.01
            t += 1
        return accuracy

if __name__ == '__main__':
    #data load
    # f = open("output.txt", "r")
    # content = f.readlines()
    torch.set_printoptions(threshold=50000)
    torch.set_printoptions(linewidth=20000)
    #cuda = torch.cuda.is_available()
    #device = torch.device('cuda' if cuda else 'cpu')
    device = 'cpu'
    #model = PacketRnn(input_size= input_size ,hidden_size = 240)
    model = PacketRnn()
    model.init_w()
    ini_w = model.features[0].weight_ih_l0.clone()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    B_RNN = B_RNNtrainer(model, bit = 128, device='cuda')
    epoch_losses= B_RNN.train_step(optimizer)

    final_w = model.features[0].weight_ih_l0.clone()


    test_B_RNN = test_RNNtrainer(model, bit=128, device='cuda')
    accuracy = test_B_RNN.test(final_w, 10000)


    print(accuracy)


    # print(ini_w-final_w)
    # print("final_w", final_w)




