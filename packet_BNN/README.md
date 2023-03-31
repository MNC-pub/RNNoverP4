# BNN for Packet classification #

Binary data used for training is packets extracted from DARPA 1998 intrusion dataset.

These packets are represented by 126bits that is 5tuple with ACK and Syn Flag, and saved in "BNN_dataset.txt".
Approximately 180,000 packets are used in dataset.

## dependency
1. This python file is tested in python 3.9 version.
2. Also, you need to have pytorch installed in python. 

### BNN Weight creation

1. change settings for weight variation.
- default learning rate and the number of training packet is 0.01 and 10000, respectively.
- currently, 1000 packets are used for training in default.

2. run python file "packet_BNN.py"
- this file will train the weight of the layers under configured settings.
- testing the trained weight will be follwed after.
- you can get precision, F1 score of BNN weight in the testing process.
- It takes time.


inspired by Q. Qin, K. Poularakis, K. K. Leung and L. Tassiulas, "Line-Speed and Scalable Intrusion Detection at the Network Edge via Federated Learning,"
2020 IFIP Networking Conference (Networking), 2020, pp. 352-360.

code modified from https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-1.0


