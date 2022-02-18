from kamene.all import *
import sys
sys.stdout = open('output.txt','w')

def len2bin(len):
    binary = format(len, '016b')
    return binary

def protocol2bin(proto):
    binary = format(proto, '08b')
    return binary

def ip2bin(ip):
    octets = map(int, ip.split('.'))
    binary = '{0:08b}{1:08b}{2:08b}{3:08b}'.format(*octets)
    return binary

def L42bin(L4):
    binary = format(L4, '016b')
    return binary

pkts = rdpcap("data.pcap")

for i in range (0,100):
    totalLen = pkts[i][IP].len
    protocol = pkts[i].proto
    srcAddr = pkts[i][IP].src
    dstAddr = pkts[i][IP].dst
    L4src = pkts[i][TCP].sport
    L4dst = pkts[i][TCP].dport

    BNNinput = len2bin(totalLen)+protocol2bin(protocol)+ip2bin(srcAddr)+ip2bin(dstAddr)+L42bin(L4src)+L42bin(L4dst)
    print(BNNinput)
