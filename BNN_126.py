#malicious한 IP로 왔는지 source를 대조
import torch
from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import linear, conv2d
import torch.nn as nn
#from torch.autograd import Variable
import torch.autograd as autograd
import sys
from matplotlib import pyplot as plt

__all__ = ['packetbnn']

class Packetbnn(nn.Module):

    def __init__(self, bit):
        super(Packetbnn, self).__init__()
        self.bit = bit
        self.features = nn.Sequential(

            BNNConv2d(1, bit, kernel_size=(1,bit), stride=1, padding=0),
            nn.GroupNorm(1,bit),
            # BNNConv2d(1, 120, kernel_size=(1, 120), stride=1, padding=0),
            # nn.GroupNorm(1, 120),
            # BNNConv2d(1, 120, kernel_size=(1, 120), stride=1, padding=0),
            # nn.GroupNorm(1, 120),
            # BNNConv2d(1, 120, kernel_size=(1, 120), stride=1, padding=0),
            # nn.GroupNorm(1, 120),
            nn.Softsign(),

            nn.Flatten(),
            BNNLinear(bit, 1),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        # weight initialization


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                # nn.init.kaiming_normal_(m.weight_org, mode='fan_out')
                nn.init.uniform_(m.weight, a= -.5, b= 0.5)
                nn.init.uniform_(m.weight_org, a=-.5, b=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a= 1., b= 1.)
                #nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        return

def packetbnn(num_classes=1):
    return Packetbnn(num_classes)

def Binarize(tensor):

    return tensor.sign()

def data_Binarize(tensor):
    # 1, -1 >> 1, 0
    return tensor.add_(1).div_(2)


# def input_Binarize(tensor):
#     # 1과 0을 -1과 1로
#     for i in range(0, tensor.size(dim=0)):
#         if tensor[i] == 0:
#             tensor[i] = -1
#
#         tensor = tensor.sign()
#     return tensor

def input_Binarize(tensor):
    # 1과 0을 1과 -1로
    result = (tensor-0.5).sign()
    return result


class BNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BNNConv2d(Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        input.data = input_Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)


        out = conv2d(input, self.weight.data, None, self.stride,
                     self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class Bnntrainer():
    def __init__(self, model, bit, lr , device=None):
        super().__init__()
        self.model = model
        self.bit = bit
        self.lr = lr
        self.device = device
        self.label = torch.zeros(180000, 1)
        self.data = torch.zeros(100000, self.bit)

    def train_step(self, optimizer):
        #data = torch.zeros(182000, self.bit)

        epoch_losses = []
        epoch_loss = 0
        input = torch.zeros(1,1,1,self.bit)
        # label = torch.zeros(180000, 1)
        # label = labeling.label()

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
                    self.label[label_Index] = torch.tensor(int(i))
                    # target[label_Index] = torch.tensor(int(i))
                    label_Index += 1

        f = open("final_0322.txt", "r")
        content = f.readlines()
        #t means packet sequence
        data_target = [[]]
        t = 0
        for line in content:
            k = 0
            if t ==50000 :
                break
            for i in line:
                if i.isdigit() == True:
                    self.data[t][k] = int(i)
                    k += 1

            input[0][0] = self.data[t]

            target = self.label[t].clone()
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

    def inference(self):

        f = open("final_0322.txt", "r")
        content = f.readlines()

        t = 0
        for line in content:
            k = 0
            if t ==60000 :
                break
            for i in line:
                if i.isdigit() == True:
                    self.data[t][k] = int(i)
                    k += 1
            t +=1


        # f = open("weight_inference_test.txt", "r")
        # content = f.readlines()
        weight = torch.zeros(self.bit +1 ,1,1,self.bit)

        accuracy = 0
        predict = torch.zeros(1)
        middle_result = torch.zeros(self.bit, self.bit)
        middle_result2 = torch.zeros(self.bit)
        middle_result3 = torch.zeros(self.bit)
        weight[0:self.bit] = Binarize(self.model.features[0].weight)
        weight[self.bit:] = Binarize(self.model.features[1].weight)
        print(weight)
        weight= data_Binarize(weight)
        print(weight)

        # print(weight[121].size())
        for t in range(50000,51000) :
            print(t)
            for k in range(0, self.bit):

                middle_result[k] = XNOR(self.data[t], weight[k][0][0], self.bit)

                # print(self.data[t])
                # print("weight",weight[k][0][0])
                # print("m_r", middle_result[k])
                middle_result2[k] = Bitcount(middle_result[k])
                # print("m_r2",middle_result2[k])

            middle_result3 = XNOR(middle_result2, weight[self.bit][0][0], self.bit)
            middle_result3 = middle_result3.type(torch.int8)
            print(middle_result3)
            predict = Bitcount(middle_result3)
            # if predict == 0:
            #     predict -= 1
            target = torch.tensor(self.label[t])
            print("predict", predict)
            print("target", target)
            if predict == target:
                accuracy += 0.1

            t += 1
        return accuracy

def XNOR(A, B, bit):
    R = torch.zeros(bit)
    for i in range(0,bit):

        if A[i] == 0 and B[i] == 0:
            R[i]= 1
        if A[i]  == 0 and B[i] == 1:
            R[i]= 0
        if A[i]  == 1 and B[i] == 0:
            R[i]= 0
        if A[i]  == 1 and B[i] == 1:
            R[i] = 1
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




if __name__ == '__main__':
    #data load
    # f = open("output.txt", "r")
    # content = f.readlines()
    torch.set_printoptions(threshold=50000)
    torch.set_printoptions(linewidth=20000)
    # cuda = torch.cuda.is_available()
    # device = torch.device('cuda' if cuda else 'cpu')
    device = torch.device('cpu')
    # print(device)
    Packetbnn = Packetbnn(bit = 126)
    # model = eval("packetbnn")()
    # model.to(device)
    Packetbnn.to(device)
    Packetbnn.init_w()

    # sample input

    Bnn = Bnntrainer(Packetbnn, bit=126, lr=0.01,device='cuda')
    optimizer = torch.optim.Adam(Packetbnn.parameters(), weight_decay=1e-7)
    # ini_weight = Packetbnn.features[0].weight.clone()
    # ini_weight_org = Packetbnn.features[0].weight_org.clone()
    epoch_losses= Bnn.train_step(optimizer)
    #accuracy = Bnn.inference()
    sys.stdout = open('weight.txt', 'w')
    # print(Binarize(Packetbnn.features[0].weight).byte())
    # print(Binarize(Packetbnn.features[4].weight).byte())
    print(Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int())
    # print(Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int())

    # A= Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int()
    # B= Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int()

    A = torch.zeros(200,126)

    f = open('weight.txt', "r")
    content = f.readlines()
    # print(Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int())

    sys.stdout = open('weight_final.txt', 'w')
    t = 0
    for line in content:
        k = 0
        if t % 3 == 0:
            for i in line:
                if i.isdigit() == True:
                    if t == 0:
                        # A[0][k] = int(i)
                        print(int(i), end='')
                        k += 1
                    else:
                        T = int(t / 3)
                        # A[T][k] = int(i)
                        print(int(i), end='')
                        k += 1
            print("")
        t += 1





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

    #sys.stdout = open('weight.txt', 'w')

    # print(Packetbnn.features[0].weight)

    # print(Binarize(Packetbnn.features[0].weight).byte())
    # print(Binarize(Packetbnn.features[4].weight).byte())
    # print(Binarize(Packetbnn.features[0].weight).add_(1).div_(2).int())
    # print(Binarize(Packetbnn.features[4].weight).add_(1).div_(2).int())
