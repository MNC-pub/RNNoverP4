#!/usr/bin/env python
import argparse
import sys
import socket
import random
import struct

from scapy.all import sendp, send, get_if_list, get_if_hwaddr
from scapy.all import Packet
from scapy.all import Ether, IP, UDP, TCP
from scapy.all import hexdump, BitField, BitFieldLenField, ShortEnumField, X3BytesField, ByteField, XByteField

class Weightwriting(Packet):

    name = "Weightwriting"

    fields_desc = [
        BitField("index", 1, 8), # index: 0~119, only need 7
        BitField("weight", 10, 40) # weight: (in decimal, largest is) 2^120 -1 = 1.1329*10^36, only need 37
    ]

def main():
    # 3 arguments needed: src / dst / veth
    if len(sys.argv)<3:
        print('pass 1 arguments: <destination> ')
        exit(1)

# src addr
    addr = socket.gethostbyname(sys.argv[1])
# dst addr
    addr1 = socket.gethostbyname(sys.argv[2])
# veth
    iface = sys.argv[3]

# read 120x120 bit weight line by line from txt
    f = open("output.txt", 'r')
    lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    w=[]

    print("sending on interface %s (Bmv2 port)" % (iface))

# send one line (120 bits converted to decimal) for 120 times
    for i in range(0, 120):
        w.append(int(lines[i], 2))
        pkt = Ether() / IP(src=addr, dst=addr1) / Weightwriting(index=i, weight=w[i]) / UDP()
        pkt.show()
        hexdump(pkt)
        sendp(pkt, iface=iface, verbose=False)

if __name__ == '__main__':
    main()

#sudo python3 ./send_bnnweight.py 10.0.0.1 10.0.0.2 veth0
#sudo python3 ./receive.py --i veth1