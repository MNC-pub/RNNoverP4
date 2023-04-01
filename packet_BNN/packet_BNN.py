import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import linear, conv2d
import os
import numpy as np
from torch import save, no_grad
from tqdm import tqdm
# import inference
import labeling
import sys


__all__ = ['bnn_caffenet']

global true_positive  # TP
global false_negative  # FN
global false_positive  # FP
global tos_count_1
global tos_count_0


class Packetbnn(nn.Module):

    def __init__(self, num_classes=2):
        super(Packetbnn, self).__init__()

        self.features = nn.Sequential(

            BNNConv2d(1, 126, kernel_size=(1,126), stride=1, padding=0, bias=False),
            nn.GroupNorm(1,126),
            nn.Hardtanh(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),

            nn.Flatten(),
            nn.GroupNorm(1,126),
            nn.Hardtanh(inplace=True),
            BNNLinear(126, num_classes),
            # nn.BatchNorm1d(num_classes, affine=False),
            nn.GroupNorm(1,2),
            #nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return


def bnn_caffenet(num_classes=1):
    return Packetbnn(num_classes)


__all__ = ['BNNLinear', 'BNNConv2d']


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    if quant_mode == 'bin':
        return (tensor >= 0).type(type(tensor)) * 2 - 1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class BNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)
        out = linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BNNConv2d(Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)

        out = conv2d(input, self.weight, None, self.stride,
                     self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

def XNOR(A, B):
    R = torch.zeros(126, dtype=torch.int64 )
    for i in range(0,126):
        if (A[i] == B[i]) :
            R[i] = 1
        else :
            R[i] = 0
    return R


def Bitcount(tensor):
    tensor = tensor.type(torch.int8)
    activation = torch.zeros(1, 1)

    count = torch.bincount(tensor)
    k = torch.tensor(63)
    # activation
    if count.size(dim=0) == 1:
        activation = torch.tensor([[0.]])
    elif count[1] > k:
        activation = torch.tensor([[1.]])
    else:
        activation = torch.tensor([[0.]])
    return activation

class BnnClassifier():
    def __init__(self, model, data, label, train_packet=None, device=None):
        super().__init__()
        self.model = model
        self.data = data
        self.label = label
        self.train_packet = train_packet
        self.device = device


    def test(self, num_test_packet):
        predict = torch.zeros(1)
        middle_result = torch.zeros(126, 126)
        middle_result2 = torch.zeros(126)
        linear_layer_result_1 = torch.zeros(126)
        linear_layer_result_2 = torch.zeros(126)
        learned_weight = torch.zeros(128,1,1, 126)
        weight = torch.zeros(128, 126)

        true_positive = 0
        false_positive = 0
        false_negative = 0
        total1 = 0
        total2 = 0
        tos_count_1 = 0
        tos_count_0 = 0
        precision = 0
        data = torch.zeros(30000, 126)
        f = open("BNN_test_dataset.txt", "r")
        content = f.readlines()
        t = 0
        for line in content:
            k = 0
            if t == 10000:
                break
            for i in line:
                if i.isdigit() == True:
                    data[t][k] = int(i)
                    k += 1
            t += 1

        target = labeling.label(30000)
        learned_weight[0:126] = Binarize(self.model.features[0].weight).add_(1).div_(2).int()
        weight[0:126] = learned_weight[0:126][0][0][:]

        weight[126:128] = Binarize(self.model.features[6].weight).add_(1).div_(2).int()

        for z in range(10000, 10000+num_test_packet):

            for k in range(0, 126):
                middle_result[k] = XNOR(data[z], weight[k])
                middle_result2[k] = Bitcount(middle_result[k])

            linear_layer_result_1 = XNOR(middle_result2, weight[126])
            linear_layer_result_2 = XNOR(middle_result2, weight[127])

            if not torch.ge(torch.bincount(linear_layer_result_1), torch.bincount(linear_layer_result_2))[1] :
                predict = 0
            else:
                predict = 1
            # print(predict)
            # print(target[z])
            target[z] = target[z].astype(np.int32)
            if predict == 1:
                if target[z] == 1:
                    true_positive = true_positive + 1
                if target[z] == 0:
                    false_positive = false_positive + 1
            else:
                if target[z] == 1:
                    false_negative = false_negative + 1
            if z %100 == 0 :
                 print("testing progress: ", (z-10000)/num_test_packet*100, "%" )

        total1 = true_positive + false_negative
        total2 = true_positive + false_positive
        if (total1 != 0 and total2 != 0):
            recall = float(true_positive) / float(total1)
            precision = float(true_positive) / float(total2)
            f1score = 2.0 / float(1 / precision + 1 / recall)
        else:
            recall = 0
            f1score = 0

        print("TP : {}, FP : {}, FN : {}, recall rate : {}, precision : {},  f1score : {}".format(
            true_positive, false_positive, false_negative, recall, precision, f1score))
        return

    def train_step(self, criterion, optimizer):
        losses = []
        input = torch.zeros(1, 1, 1, 126)

        for t in range(self.train_packet) :
            input[0][0] = self.data[t]
            output = self.model(input)
            target = torch.tensor([self.label[t]], dtype=torch.int64)
            loss = criterion(output, target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight.data.copy_(p.weight_org)
            optimizer.step()
            for p in self.model.modules():
                if hasattr(p, 'weight_org'):
                    p.weight_org.data.copy_(p.weight.data.clamp_(-1, 1))
            if t %1000 == 0 :
                print("learning progress: ", t/100, "%" )
        return losses

    def train(self, criterion, optimizer, epochs, num_test_packet):

        losses = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_losses = self.train_step(criterion, optimizer)
            losses += epoch_losses
            epoch_losses = np.array(epoch_losses)
            lr = optimizer.param_groups[0]['lr']
            self.test(num_test_packet)
            print('Train Epoch {0}\t Loss: {1:.6f}\t lr: {2:.4f}'
                   .format(epoch, epoch_losses.mean(), lr))
            # print('Train Epoch {0}\t Loss: {1:.6f}\t Test Accuracy {2:.3f} \t lr: {3:.4f}'
            #        .format(epoch, epoch_losses.mean(), test_accuracy, lr))
            # print('epoch losses: {:.3f} '.format(best_accuracy))
        return


torch.set_printoptions(threshold=50000)
torch.set_printoptions(linewidth=20000)

# cuda = torch.cuda.is_available()
# device = torch.device('cuda' if cuda else 'cpu')
device = 'cpu'

model = Packetbnn()
model.to(device)

learning_packets = 10000
packet_dataset_size= 30000
label = labeling.label(packet_dataset_size)

data = torch.zeros(packet_dataset_size, 126)
f = open("BNN_test_dataset.txt", "r")
content = f.readlines()
data_target = [[]]
t= 0
for line in content:
    k = 0
    if t == (packet_dataset_size):
        break
    for i in line:
        if i.isdigit() == True:
            if i == "0":
                data[t][k] = int(-1)
            else :
                data[t][k] = int(i)
            k += 1
    t += 1

classification = BnnClassifier(model,data, label, train_packet= learning_packets, device= device)

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)

if hasattr(model, 'init_w'):
    model.init_w()

optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-7)

classification.train(criterion, optimizer, epochs = 1, num_test_packet= 1000)

#
# sys.stdout = open('weight.txt', 'w')
# print(Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int())
# #packet bnn nnlayer 중 linear layer의 weight를 의미
# print(Binarize(Packetbnn.features[6].weight).add_(1).div_(2).int())
