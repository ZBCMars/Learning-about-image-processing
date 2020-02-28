from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import cv2
#import random
import math
#import heapq
from zigzag import *
import copy
import re
import struct
import os
import sys

#获取图片
def getimg():
  return Image.open("3.jpg")
  
#显示图片
def showimg(img, isgray=False):
  plt.axis("off")
  if isgray == True:
    plt.imshow(img, cmap='gray')
  else: 
    plt.imshow(img)
  plt.show()


def read_image(filename, mode="L"):
    img = Image.open(filename).convert(mode)
    pixels = np.array(img.getchannel(0))
    return pixels, img.height, img.width

def RLE(sample_flatten):
    result  = []
    counter =  1
    N = len(sample_flatten)
    prev = sample_flatten[0]    
    fomater = lambda c, v : "{0}{1}".format(c, format(v, "02x"))  
    for i in range(1, N):
        current = sample_flatten[i]
        if current == prev:
            counter += 1
        else:
            result.append(fomater(counter, prev))
            counter = 1
            prev = current
            
        if i == N - 1:
            result.append(fomater(counter, prev))

    return np.array(result)

class TreeNode(object):
    def __init__(self, data):
        self.val = data[0]
        self.priority = data[1]
        self.leftChild = None
        self.rightChild = None
        self.code = ""

def creatnodeQ(codes):
    q = []
    for code in codes:
        q.append(TreeNode(code))
    return q

def addQ(queue, nodeNew):
    if len(queue) == 0:
        return [nodeNew]
    for i in range(len(queue)):
        if queue[i].priority >= nodeNew.priority:
            return queue[:i] + [nodeNew] + queue[i:]
    return queue + [nodeNew]

class nodeQeuen(object):

    def __init__(self, code):
        self.que = creatnodeQ(code)
        self.size = len(self.que)

    def addNode(self,node):
        self.que = addQ(self.que, node)
        self.size += 1

    def popNode(self):
        self.size -= 1
        return self.que.pop(0)

def Imgpoint(image):
    weight = {}
    
    binfile = open(image, 'rb') 
    size = os.path.getsize(image) 
    for i in range(size):
        data = binfile.read(1)
        temp = int.from_bytes(data,byteorder='big',signed=False)
        if (temp < 128) and (temp>0):
            data = str(hex(ord(data)))
            key = data.split("x")[1]
            if (len(key)==1):
                key = '0'+key
        else:
            key = str(data).split("x")[1][0:-1]

        if not key in weight:
            weight[key] = 1
        else:
            weight[key]+= 1

    binfile.close()
    results = sorted(weight.items(),key=lambda x:x[1])
    return results
def creatHuffman(nodeQ):
    while nodeQ.size != 1:
        node1 = nodeQ.popNode()
        node2 = nodeQ.popNode()
        r = TreeNode([None, node1.priority+node2.priority])
        r.leftChild = node1
        r.rightChild = node2
        nodeQ.addNode(r)
    return nodeQ.popNode()

def HuffmanDic(head, x, Dic1, Dic2):
    if head:
        HuffmanDic(head.leftChild, x+'0',Dic1,Dic2)
        head.code += x
        if head.val:
            Dic2[head.code] = head.val
            Dic1[head.val] = head.code
        HuffmanDic(head.rightChild, x+'1',Dic1,Dic2)
    return (Dic1,Dic2)

def binaryToHex(binaryString):
    return str(hex(int(binaryString,2))) 

def Encode(image,Dic1):

    binfile = open(image, 'rb') 
    size = os.path.getsize(image) 
    transcode = ""
    hexCode = ""
    
    for i in range(size):
        data = binfile.read(1)
        temp = int.from_bytes(data,byteorder='big',signed=False)
        if (temp < 128) and (temp>0):
            data = str(hex(ord(data)))
            key = data.split("x")[1]
            if (len(key)==1):
                key = '0'+key
            transcode += Dic1[key]
        else:
            key = str(data).split("x")[1][0:-1]
            transcode += Dic1[key]
    binfile.close()
    length = len(transcode)

    left = length % 4 
    for i in range(left):
        transcode = '0' + transcode
    
    if((int(len(transcode)/4) % 2) == 1):
        transcode = '0000' + transcode
        left = left + 4
    
    for i in range(0,int(len(transcode)),8):
        slice = transcode[i:i+8]
        a = str(binaryToHex(slice)).split('x')[1]
        if len(a) < 2:
            a = '0'+a
        hexCode += a
    hexCode = bytearray.fromhex(hexCode)
    return (hexCode,left)

def getDict(image):
    Dic1 = {}
    Dic2 = {}
    weights = Imgpoint(image)
    t = nodeQeuen(weights)
    tree = creatHuffman(t)
    Dic1,Dic2= HuffmanDic(tree, '',Dic1,Dic2)
    return (Dic1,Dic2)

def Compression(image,Dic1):
    huffman = Encode(image,Dic1)
    return huffman



#img = getimg()
#img = cv2.imread("3.jpg")
#Rimg, Gimg, Bimg = cv2.split(img)


#homework1--------------------------------------------------------------------------------------------
#question1--------------------------------------------------------------------------------------------
'''pixels, h, w  = read_image("2.jpg")
sample_flatten = pixels.flatten()
encoded = RLE(sample_flatten)
print("\nLength After encoding:", len(encoded))
print("Ratio: {0}%".format(round(100 * len(encoded) / len(sample_flatten), 4)))'''

#question2--------------------------------------------------------------------------------------------
image = "4.jpg"
fp = open(image,'rb')
content = fp.read()
content_size = sys.getsizeof(content)
fp.close()
dict1,dict2 = getDict(image)
huffman,left = Compression(image,dict1)
huffman_size = sys.getsizeof(huffman)
huffman_ratio = content_size/huffman_size
print("length before coding:" + str(content_size))
print("length after coding:" + str(huffman_size))
print("ratio:" + str(huffman_ratio))