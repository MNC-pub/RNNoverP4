BNN for Packet classification(incomplete)

inspired by Q. Qin, K. Poularakis, K. K. Leung and L. Tassiulas, "Line-Speed and Scalable Intrusion Detection at the Network Edge via Federated Learning,"
2020 IFIP Networking Conference (Networking), 2020, pp. 352-360.


code reused from https://github.com/lucamocerino/Binary-Neural-Networks-PyTorch-1.0

packet_BNN.py is main file in this code.
you will only need labeling.py to run this packet_BNN.py which is used for labeling malicious packet.

Binary data is 5 tuple bitdata extracted from ISCX BOTNET2014.

remaining problem - gradient is not working as well.
all Weight component goes to 0 when learning 
