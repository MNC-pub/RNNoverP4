import torch
import labeling
from torch.nn.functional import linear, conv2d



def XNOR(A, B):
    R = torch.zeros(120)
    for i in range(0,120):

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
    k = torch.tensor(60)
    # activation
    if count.size(dim=0) == 1:
        activation = torch.tensor([[0.]])
    elif count[1] > k:
        activation = torch.tensor([[1.]])
    else:
        activation = torch.tensor([[0.]])
    return activation

def inference():
    data = torch.zeros(12000,120)
    label = labeling.label()

    f = open("output_test.txt", "r")
    content = f.readlines()
    # t means packet sequence

    t = 0
    for line in content:
        k = 0
        for i in line:
            if i.isdigit() == True:
                data[t][k] = int(i)
                k += 1
        t += 1

    f = open("weight_inference_test.txt", "r")
    content = f.readlines()
    weight = torch.zeros(121,120)

    t = 0
    for line in content:
        k = 0
        if t%3 == 0:
            for i in line:
                if i.isdigit() == True:
                    if t ==0 :
                        weight[0][k] = int(i)
                        k += 1
                    else :
                        T = int(t/3)
                        weight[T][k] = int(i)
                        k += 1

        t += 1

    torch.set_printoptions(threshold=50000)
    torch.set_printoptions(linewidth=20000)

    print(weight)
    accuracy = 0
    predict = torch.zeros(1)
    middle_result = torch.zeros(120,120)
    middle_result2 = torch.zeros(120)
    middle_result3 = torch.zeros(120)
    label = labeling.label()
    for z in range(5000,5300) :
        print(z)
        #for k in range(0,120) :
        for k in range(0, 120):
            middle_result[k] = XNOR(data[z] , weight[k])
            #print(middle_result[k])
            middle_result2[k] = Bitcount(middle_result[k])
            #print(middle_result2[k])
        middle_result3 = middle_result2 * weight[120]
        #print(middle_result3)
        predict = Bitcount(middle_result3)
        target = torch.tensor(label[z])
        print("predict",predict )
        print("target", target)
        if predict == target :
            accuracy+= 1
    return accuracy

print(inference())
