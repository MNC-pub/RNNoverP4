import torch

import labeling
import sys

global true_positive  # TP
global false_negative  # FN
global false_positive  # FP
global tos_count_1
global tos_count_0


# def XNOR(A, B):
#     R = torch.zeros(126)
#     for i in range(0,126):
#
#         if A[i] == 0 and B[i] == 0:
#             R[i]= 1
#         if A[i]  == 0 and B[i] == 1:
#             R[i]= 0
#         if A[i]  == 1 and B[i] == 0:
#             R[i]= 0
#         if A[i]  == 1 and B[i] == 1:
#             R[i] = 1
#     return R

def XNOR(A, B):
    R = torch.zeros(126)
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

def inference(packets):
    data = torch.zeros(11000,126)
    label = torch.zeros(20001, 1)

    f = open("BNN_dataset.txt", "r")
    content = f.readlines()
    data_Index = 0
    q = 0
    for line in content:
        w = 0
        if q == 10000:
            break
        for i in line:
            if i.isdigit() == True:
                data[q][w] = int(i)
                w += 1
        q += 1

    target = labeling.label()
    weight = torch.zeros(127,126)

    f = open("weight_final_bnn.txt", "r")
    content = f.readlines()
    # t means packet sequence
    t = 0
    for line in content:
        k = 0
        for i in line:
            if i.isdigit() == True:
                weight[t][k] = int(i)
                k += 1
        t += 1
    # torch.set_printoptions(threshold=50000)
    # torch.set_printoptions(linewidth=20000)

    #weight 제대로 들어갔는지 확인하느 ㄴ코드
    # sys.stdout = open('bnn_weight.txt', 'w')
    # for t in range(126):
    #     print(weight[t],sep='\n')

    predict = torch.zeros(1)
    middle_result = torch.zeros(126,126)
    middle_result2 = torch.zeros(126)
    middle_result3 = torch.zeros(126)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    total1 = 0
    total2 = 0
    tos_count_1 = 0
    tos_count_0 = 0

    for z in range(10000,10000+packets) :
        print(z-10000)

        for k in range(0, 126):
            middle_result[k] = XNOR(data[z] , weight[k])

            middle_result2[k] = Bitcount(middle_result[k])
        # print(middle_result)
        # print(middle_result2)
        middle_result3 = XNOR(middle_result2, weight[126])
        middle_result3 = middle_result3.type(torch.int8)
        # print(middle_result3)
        predict =Bitcount(middle_result3)
        predict = predict.type(torch.int8)
        #print(predict)
        #print(target[z])
        if predict == 0 :
            if target[z] == -1 :
                true_positive = true_positive + 1
            if target[z] == 1 :
                false_positive = false_positive + 1
        else :
            if target[z] == -1:
                false_negative = false_negative + 1

    total1 = true_positive + false_negative
    total2 = true_positive + false_positive
    if (total1 != 0 and total2 != 0):
        recall = float(true_positive) / float(total1)
        precision = float(true_positive) / float(total2)
        f1score = 2.0 / float(1 / precision + 1 / recall)
    else:
        recall = 0
        f1score = 0

    print("TP : {}, FP : {}, FN : {}, recall rate : {},  f1score : {}".format(
        true_positive, false_positive, false_negative, recall, f1score))


inference(packets=1000)
