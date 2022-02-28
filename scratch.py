import numpy as np
import torch
from torch.nn import Module, RNN, Linear
from torch.nn.functional import linear, conv2d, hardtanh
import torch.nn as nn
#from torch.autograd import Variable
import torch.autograd as autograd
import sys
from matplotlib import pyplot as plt
sequence_length = 1
input_size = 10
hidden_size = 100
num_layers = 1
num_classes = 2
batch_size = 10
learning_rate = 0.01

def Binarize(tensor):
    #result = (tensor-0.5).sign()
    return tensor.sign()

def input_Binarize(tensor):
    return tensor.sub_(0.5).sign()

class B_RNN(RNN) :
    def __init__(self, *kargs, **kwargs):
        super(B_RNN, self).__init__(*kargs, **kwargs)
        self.input_size  = 10
        self.hidden_size = 240
        self.batch_size = 10
        self.n_layers = 1
    #weight initialize
        self.h_t = torch.autograd.Variable(torch.zeros(1, self.hidden_size))
        self.weight_ih = torch.randn((), device='cpu', dtype=torch.float, requires_grad=True)
        self.weight_hh = torch.randn((), device='cpu', dtype=torch.float, requires_grad=True)
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

        self.weight_ih.data = Binarize(self.weight_ih_org)
        self.weight_hh.data = Binarize(self.weight_hh_org)

        for i in range(self.batch_size):

            data_ih = torch.dot(input[i], self.weight_ih)
            data_hh = torch.dot(self.h_t, self.weight_hh)
            middle = data_ih + data_hh
            #ste function?
            self.h_t = torch.sign(middle)
        # output = nn.sign(middle)

        return self.h_t

B_RNN = B_RNN(input_size = 10, hidden_size = 100)

input = torch.randn(1,10,10)

output = B_RNN.forward(input)
print(output)

