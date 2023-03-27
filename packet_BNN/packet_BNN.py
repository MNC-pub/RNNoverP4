import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear
from torch.nn.functional import linear, conv2d
from torch.autograd import Variable
import labeling
class Packetbnn(nn.Module):

    def __init__(self, num_classes=1):
        super(Packetbnn, self).__init__()

        self.features = nn.Sequential(

            BNNConv2d(1, 120, kernel_size=(1,120), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(120),
            nn.Softsign(),

            nn.Flatten(),
            nn.BatchNorm1d(120),
            BNNLinear(120, 1),
            nn.BatchNorm1d(num_classes, affine=False),
            #nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.features(x)

    def init_w(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.uniform_(m.weight, a= 0., b= 1.)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.5, 0.01)
                nn.init.zeros_(m.bias)
        return


__all__ = ['BNNLinear', 'BNNConv2d']


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        result = (tensor-0.5).sign().add_(1).div_(2)
        return result
    if quant_mode == 'bin':
        return (tensor >= 0).type(type(tensor)) * 2 - 1
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


class BNNLinear(Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        # #if (input.size(1) != 784) and (input.size(1) != 3072):
        # input.data = Binarize(input.data)

        # self.weight.data = Binarize(self.weight_org)
        out = linear(input, self.weight.data)

        # if not self.bias is None:
        #     self.bias.org = self.bias.data.clone()
        #     out += self.bias.view(1, -1).expand_as(out)

        return out


class BNNConv2d(Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConv2d, self).__init__(*kargs, **kwargs)
        self.register_buffer('weight_org', self.weight.data.clone())

    def forward(self, input):

        # input.data = Binarize(input.data)
        #
        # self.weight.data = Binarize(self.weight_org)

        out = conv2d(input, self.weight.data, None, self.stride,
                     self.padding, self.dilation, self.groups)

        # if not self.bias is None:
        #     self.bias.org = self.bias.data.clone()
        #     out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


torch.set_printoptions(threshold=50000)
torch.set_printoptions(linewidth=20000)
#(1,5) 크기의 3개의 패킷
losses = []
label = labeling.label()
#input = torch.randn((2,1,1,5), requires_grad=True)
# input = torch.tensor([[[[0., 0., 1., 1., 0.]]],
#
#                       [[[1., 0., 0., 0., 0.]]],
#
#                       [[[1., 0., 0., 1., 1.]]]])
# input = torch.tensor([[[[0., 0., 1., 1., 0.]]]])
bnn = Packetbnn()
bnn.init_w()

data = torch.zeros(8000, 120)
losses = []
input = torch.zeros(1, 1, 1, 120)
f = open("output_test.txt", "r")
content = f.readlines()
# t means packet sequence
optimizer = torch.optim.Adam(bnn.parameters(), lr=0.005, weight_decay=1e-5)
t = 0
label = labeling.label()
for line in content:
    k= 0
    for i in line:
        if i.isdigit() == True:
            data[t][k] = int(i)
            k += 1
    t += 1
    input[0][0] = data[t]
    output = bnn(input)
    target = torch.tensor(label[t])
    loss = (output-target).pow(2).sum()
    loss = Variable(loss, requires_grad=True)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    for p in bnn.modules():
        if hasattr(p, 'weight_org'):
            p.weight.data.copy_(p.weight_org)
    optimizer.step()
    for p in bnn.modules():
        if hasattr(p, 'weight_org'):
            p.weight_org.data.copy_(p.weight.data.clamp_(-1, 2))
    if t%10 ==0 :
        print(losses[t])


# print(k.features[0].weight)
print(bnn.features[0].weight)
print(bnn.features[2].weight)

print(Binarize(bnn.features[0].weight))
print(Binarize(bnn.features[2].weight))
print(torch.cuda.is_available())
# a = torch.tensor([[[[-0.1980,  0.0351, -0.4719,  0.6720,  1.2105]]]])
# filter = torch.tensor([[[[-1.0145,  0.0434,  0.1013,  0.2565,  0.1284]]],
#
#
#         [[[-0.4041,  0.2468, -0.0881, -0.0285, -0.0488]]],
#
#
#         [[[-0.3031, -0.1127, -0.2771, -0.2324, -0.0808]]],
#
#
#         [[[-0.1816,  0.2063, -0.1439,  0.0060, -0.1139]]],
#
#
#         [[[-0.5576, -0.0296,  0.0776,  0.6105,  0.0445]]]], requires_grad=True)
#
# out = conv2d(a, filter, None)
# out = linear(input, self.weight)
# print(out)



# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# print(input)
# learning_rate = 1e-6
# target = torch.empty(3, dtype=torch.long).random_(5)
# print(target)

# for t in range(2000):
#
#     output = loss(input, target)
#     if t % 100 == 1:
#         print(t, output)
#
#     output.backward()
#     with torch.no_grad():
#         input -= learning_rate * input.grad
#     input.grad = None
#
# print(input)
