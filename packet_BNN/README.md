# BNN for Packet classification#

Binary data used for training is packets extracted from DARPA 1998 intrusiong dataset.

These packets are represented by 126bits that is 5tuple with ACK and Syn Flag, and saved in "BNN_dataset.txt".
Approximately 180,000 packets are used in dataset.

## dependency
1. This python file is tested in python 3.9 version.
2. Also, you need to have pytorch install in python. 

### BNN Weight creation

1. change settings for weight variation.
- change learning rate, training packet data size in line 191.
- 20,000 packets are used for training in default.

2. run python file "packet_BNN.py"
- weights of Conv2d layer and linear layer are extracted in "weight.txt"
- "weight.txt" needs to be preprocessed since it contains unneccesary characters.
- "weight_final_bnn.txt" is the final result.

3. open "weight_final_bnn.txt" and delete "32" in last line

4. run python file "inference.py"
- you can get precision, F1 score of BNN weight inference by running this file.
- in line 144, you can change the number of packets to be inference.
- It takes time.




inspired by Q. Qin, K. Poularakis, K. K. Leung and L. Tassiulas, "Line-Speed and Scalable Intrusion Detection at the Network Edge via Federated Learning,"
2020 IFIP Networking Conference (Networking), 2020, pp. 352-360.

code reused from https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-1.0


