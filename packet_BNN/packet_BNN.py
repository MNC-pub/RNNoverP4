#malicious한 IP로 왔는지 source를 대조
import torch
from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import linear, conv2d
import labeling
import torch.nn as nn
#from torch.autograd import Variable
import torch.autograd as autograd
import sys
from matplotlib import pyplot as plt

__all__ = ['packetbnn']

class Packetbnn(nn.Module):

    def __init__(self, num_classes=1):
        super(Packetbnn, self).__init__()

        self.features = nn.Sequential(

            BNNConv2d(1, 120, kernel_size=(1,120), stride=1, padding=0),
            nn.GroupNorm(1,120),
            nn.Softsign(),

            nn.Flatten(),
            BNNLinear(120, 1),

        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        # weight initialization


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.uniform_(m.weight, a= .0, b= 0.7)
                nn.init.uniform_(m.weight_org, a=.0, b=0.7)
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
                nn.init.zeros_(m.bias)
        return

def packetbnn(num_classes=1):
    return Packetbnn(num_classes)

def Binarize(tensor):
    result = (tensor-0.5).sign().add_(1).div_(2)
    return result


# class STEFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         return (input > 0).float()
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


class BNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        out = linear(input, self.weight)

        return out


class BNNConv2d(Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        input.data = Binarize(input.data)

        self.weight.data = Binarize(self.weight_org)


        out = conv2d(input, self.weight.data, None, self.stride,
                     self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class Bnntrainer():
    def __init__(self, model, bit, lr=0.01, device=None):
        super().__init__()
        self.model = model
        self.bit = bit
        self.lr = lr
        self.device = device

    def train_step(self, optimizer):
        #data = torch.zeros(182000, self.bit)
        data = torch.zeros(50000, self.bit)
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
            if t ==50000 :
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
                    p.weight_org.copy_(p.weight.data.clamp_(-1, 2))
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
    optimizer = torch.optim.Adam(Packetbnn.parameters(), lr=0.01, weight_decay=1e-7)
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
    # final_weight = Packetbnn.features[0].weight
    # final_weight_org = Packetbnn.features[0].weight_org

    # print(ini_weight[0])
    # print(final_weight[0])
    # print(ini_weight[0]-final_weight[0])
    #
    # print(ini_weight_org[0])
    # print(final_weight_org[0])
    # print(ini_weight_org[0]-final_weight_org[0])

    sys.stdout = open('weight.txt', 'w')

    print(Packetbnn.features[0].weight)
    #print(Binarize(Packetbnn.features[0].weight).detach().numpy().x)
    print(Binarize(Packetbnn.features[0].weight).byte())
    print(Binarize(Packetbnn.features[4].weight).byte())
    # for i in range(40000):
    #     if i%100 == 0 :
    #         print(losses[i])

    # target = data load
