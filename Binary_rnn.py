import numpy as np
import torch
from torch.nn import Module, RNN, Linear
from torch.nn.functional import linear, conv2d, hardtanh
import torch.nn as nn
#from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import sys
from matplotlib import pyplot as plt

sequence_length = 1
input_size = 128
#128
hidden_size = 240
# num_layers = 1
num_classes = 2
batch_size = 1
learning_rate = 0.0001

def Binarize(tensor):
    # 결과물을 1과 -1 로 표현
    return tensor.sign()

def input_Binarize(tensor):
    # 들어오는 input 1 과 0을 1과 -1로 return
    return tensor.sub_(0.5).sign()

class PacketRnn(nn.Module):

    def __init__(self, num_classes=2):
        super(PacketRnn, self).__init__()
        self.Binarize = 1
        self.features = nn.Sequential(
            B_RNN(input_size,hidden_size),
            StraightThroughEstimator(),
            RNNLinear(hidden_size, 1),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        for m in self.modules():
            if isinstance(m, RNN):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')

                # nn.init.kaiming_normal_(m.weight_ih_l0, mode='fan_out')
                # nn.init.kaiming_normal_(m.weight_hh_l0, mode='fan_out')
                # nn.init.kaiming_normal_(m.weight_ih_l0_org, mode='fan_out')
                # nn.init.kaiming_normal_(m.weight_hh_l0_org, mode='fan_out')

                nn.init.uniform_(m.weight_ih_l0, a=-.5, b=.8)
                nn.init.uniform_(m.weight_hh_l0,a=-.5, b=.8)
                nn.init.uniform_(m.weight_ih_l0_org, a=-.5, b=.8)
                nn.init.uniform_(m.weight_hh_l0_org, a=-.5, b=.8)


            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a= 1., b= 1.)
                nn.init.zeros_(m.bias)
        return

class realisticRnn(nn.Module):

    def __init__(self, num_classes=2):
        super(realisticRnn, self).__init__()
        self.Binarize = 0
        self.features = nn.Sequential(
            RNN(input_size,hidden_size),
            RNNLinear(hidden_size, 1),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        for m in self.modules():
            if isinstance(m, RNN):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #m.h_t = torch.autograd.Variable(torch.zeros(10, batch_size, hidden_size))
                # nn.init.kaiming_normal_(m.weight_ih_l0, mode='fan_out')
                # nn.init.kaiming_normal_(m.weight_hh_l0, mode='fan_out')
                nn.init.uniform_(m.weight_ih_l0, a=-0.5, b=.5)
                nn.init.uniform_(m.weight_hh_l0, a=-0.5, b=.5)

            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a= 1., b= 1.)
                nn.init.zeros_(m.bias)
        return

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x

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

        data_ih = torch.zeros(1, hidden_size)
        data_hh = torch.zeros(1, hidden_size)
        middle = torch.zeros(1, hidden_size)
        self.h_t = torch.autograd.Variable(torch.zeros(11, batch_size, hidden_size))

        # 1과 0이던 input을 1과 -1인 형식으로 변경
        input = input_Binarize(input)
        #self.h_t[0] = input_Binarize(h_0)

        self.weight_ih_l0.data = Binarize(self.weight_ih_l0_org)
        self.weight_hh_l0.data = Binarize(self.weight_hh_l0_org)
        #print("ih",self.weight_ih_l0.size())
        #print("hh",self.weight_hh_l0.size())
        for i in range(0,10):
            #input= torch.transpose(input[0][i], 0, 1)
            data_ih = torch.matmul(self.weight_ih_l0, input[0][i])
            data_hh = torch.matmul(self.h_t[i], self.weight_hh_l0)
            middle = data_ih + data_hh
            self.h_t[i + 1] = middle
            #ste function?
            #self.h_t[i+1] = torch.sign(middle)
        # output = nn.sign(middle)

        return self.h_t[10]

class RNN(RNN) :
    def __init__(self, *kargs, **kwargs):
        super(RNN, self).__init__(*kargs, **kwargs)
        #self.batch_size = 1
        self.n_layers = 1
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

    #weight initialize
        self.weight_ih = torch.randn((), device=device, dtype=torch.float, requires_grad=True)
        self.weight_hh = torch.randn((), device=device, dtype=torch.float, requires_grad=True)


    def forward(self, input):
        # for self.h_t == None :
        #h_0 = torch.autograd.Variable(torch.zeros(1,10, hidden_size))

        # RNN = nn.RNN(input_size=input_size, hidden_size= hidden_size)
        # output, hn = RNN(input)
        data_ih = torch.zeros(1, hidden_size)
        data_hh = torch.zeros(1, hidden_size)
        middle = torch.zeros(1, hidden_size)
        self.h_t = torch.autograd.Variable(torch.zeros(11, batch_size, hidden_size))

        # 1과 0이던 input을 1과 -1인 형식으로 변경

        for i in range(0,10):
            data_ih = torch.matmul(input[0][i], self.weight_ih_l0)
            data_hh = torch.matmul(self.h_t[i], self.weight_hh_l0)
            middle = data_ih + data_hh
            #ste function?
            self.h_t[i+1] = torch.sign(middle)

        # return hn[0][9]
        return self.h_t[10]

class B_RNNtrainer():
    def __init__(self, model, bit= 128, lr= learning_rate, device=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.bit = bit
        self.device = device

    def train_step(self,optimizer):
        #data = torch.zeros(182000, self.bit)
        data = torch.zeros(100000, self.bit)
        target = torch.zeros(180000, 1)

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
            if t ==60000 :
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
                if self.model.Binarize == 1 :
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
    def __init__(self, model):
        super().__init__()
        self.model = model

    def test(self, test_start) :
        #print(self.model.features[0].weight_ih_l0)
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

            if t == test_start+4000:
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
                    #print(output)
                    Binary_output = output.sign().div_(2).add_(0.5)
                    #print(Binary_output)
                    # print("Binary_output", output)
                    # print("target[t]", target[t])
                    if int(Binary_output) == target[t] :

                        accuracy +=0.025

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
    RNN_model = realisticRnn()
    model.init_w()
    RNN_model.init_w()
    #ini_w = model.features[0].weight_ih_l0.clone()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    B_RNN = B_RNNtrainer(model, bit = 128, device='cuda')
    RNN = B_RNNtrainer(RNN_model, bit=128, device='cuda')


    B_epoch_losses= B_RNN.train_step(optimizer)
    #epoch_losses = RNN.train_step(optimizer)

    # final_w_ih = model.features[0].weight_ih_l0.clone()
    # final_w_hh = model.features[0].weight_hh_l0.clone()

    RNN_final_w_ih = RNN_model.features[0].weight_ih_l0.clone()
    RNN_final_w_hh = RNN_model.features[0].weight_hh_l0.clone()

    test_B_RNN = test_RNNtrainer(model)
    #test_RNN = test_RNNtrainer(RNN_model)


    # for i in range(3) :
    accuracy = test_B_RNN.test(60000)
        # accuracies.append(accuracy)
    #RNN_accuracy = test_RNN.test(60000)
    #
    print("accuracy",accuracy)
    #print("RNN_accuracy",RNN_accuracy)

    # print(ini_w-final_w)
    # print("final_w", final_w)


