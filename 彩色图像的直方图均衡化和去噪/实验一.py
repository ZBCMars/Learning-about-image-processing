from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

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

def image_hist(image):
    color = {"blue", "green", "red"}    
    for i, color in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

def histeq(imarr):
  hist, bins = np.histogram(imarr, 255)
  cdf = np.cumsum(hist)
  cdf = 255 * (cdf/cdf[-1])
  res = np.interp(imarr.flatten(), bins[:-1], cdf)
  res = res.reshape(imarr.shape)
  return res, hist

#作业一问题1_RGB均衡化
def rgb_histeq1(im):
  imarr = np.array(im)
  imarr2 = imarr.flatten()
  #print(imarr2)
  hist, bins = np.histogram(imarr2, 255)
  #print(bins[:-1])
  cdf = np.cumsum(hist)
  #print(cdf)
  cdf = 255* (cdf/cdf[-1])
  #print(cdf)
  imarr3 = np.interp(imarr2, bins[:-1], cdf)
  imarr3 = imarr3.reshape(imarr.shape)
  return Image.fromarray(imarr3.astype('uint8'), mode='RGB')

def RGB2HSI(rgb_img):
    row = np.shape(rgb_img)[0]
    col = np.shape(rgb_img)[1]
    hsi_img = rgb_img.copy()
    print(type(rgb_img))
    B,G,R = cv2.split(rgb_img)
    [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
    H = np.zeros((row, col)) 
    I = (R + G + B) / 3.0     
    S = np.zeros((row,col))      
    for i in range(row):
        den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
        if den.any == 0:
            continue
        thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   
        h = np.zeros(col)               
        #den>0且G>=B的元素h赋值为thetha
        h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
        #den>0且G<=B的元素h赋值为thetha
        h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
        #den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h/(2*np.pi)      #弧度化后赋值给H通道
    #计算S通道
    for i in range(row):
        min = []
        #找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j],G[i][j],R[i][j]]
            min.append(np.min(arr))
        min = np.array(min)
        S[i] = 1 - min*3/(R[i]+B[i]+G[i])
        #I为0的值直接赋值0
        S[i][R[i]+B[i]+G[i] == 0] = 0
    hsi_img[:,:,0] = H*255
    hsi_img[:,:,1] = S*255
    hsi_img[:,:,2] = I*255
    return hsi_img

def HSI2RGB(hsi_img):
    row = np.shape(hsi_img)[0]
    col = np.shape(hsi_img)[1]
    rgb_img = hsi_img.copy()
    H,S,I = cv2.split(hsi_img)
    [H,S,I] = [ i/ 255.0 for i in ([H,S,I])]
    R,G,B = H,S,I
    for i in range(row):
        h = H[i]*2*np.pi
        #H大于等于0小于120度时
        a1 = h >=0
        a2 = h < 2*np.pi/3
        a = a1 & a2         
        tmp = np.cos(np.pi / 3 - h)
        b = I[i] * (1 - S[i])
        r = I[i]*(1+S[i]*np.cos(h)/tmp)
        g = 3*I[i]-r-b
        B[i][a] = b[a]
        R[i][a] = r[a]
        G[i][a] = g[a]
        #H大于等于120度小于240度
        a1 = h >= 2*np.pi/3
        a2 = h < 4*np.pi/3
        a = a1 & a2         
        tmp = np.cos(np.pi - h)
        r = I[i] * (1 - S[i])
        g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
        b = 3 * I[i] - r - g
        R[i][a] = r[a]
        G[i][a] = g[a]
        B[i][a] = b[a]
        a1 = h >= 4 * np.pi / 3
        a2 = h < 2 * np.pi
        a = a1 & a2            
        tmp = np.cos(5 * np.pi / 3 - h)
        g = I[i] * (1-S[i])
        b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
        r = 3 * I[i] - g - b
        B[i][a] = b[a]
        G[i][a] = g[a]
        R[i][a] = r[a]
    rgb_img[:,:,0] = B*255
    rgb_img[:,:,1] = G*255
    rgb_img[:,:,2] = R*255
    return rgb_img

def jiaoyan_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def gas_noise(image, mean=0, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

'''def junzhi_filter(image, size):
    input_image = np.copy(image)  # 输入图像的副本
    template = np.ones((size, size))  # 空间滤波器模板
    num = int((size - 1) / 2)  # 输入图像需要填充的尺寸
    #input_image = np.pad(input_image, (num, num), mode="constant", constant_values=0)  # 填充输入图像
    m = input_image.shape[0]
    n = input_image.shape[1]  
    output_image = np.copy(input_image)  # 输出图像
    #i = num
    #j = num
    print(input_image[1, 1])
    #k = input_image[i - num:i + num + 1, j - num:j + num + 1]
    #k = np.multiply(template, input_image[i - num:i + num + 1, j - num:j + num + 1])
    #k = np.sum(np.multiply(template, input_image[i - num:i + num + 1, j - num:j + num + 1]))# / (size ** 2)   
    #print(k)
    # 空间滤波
    for i in range(num, m - num):
        for j in range(num, n - num):
            B= input_image[i, j][0]
            G= input_image[i, j][1]
            R= input_image[i, j][2]

            #output_image[i, j] = np.sum(np.multiply(template, input_image[i - num:i + num + 1, j - num:j + num + 1])) / (size ** 2) 
    B = np.zeros(m, n)        
    for i in range(num, m - num):
        for j in range(num, n - num):
            B[i, j] = np.sum(np.multiply(template, B[i - num:i + num + 1, j - num:j + num + 1])) / (size ** 2)
            G[i, j] = np.sum(np.multiply(template, G[i - num:i + num + 1, j - num:j + num + 1])) / (size ** 2)
            R[i, j] = np.sum(np.multiply(template, R[i - num:i + num + 1, j - num:j + num + 1])) / (size ** 2)
            output_image[i, j] = cv2.merge((R, G, B))
    output_image = output_image[num:m - num, num:n - num]  # 裁剪
    return output_image'''


def junzhi(image, size):
    input_image = np.copy(image)  
    filter_template = np.ones((size, size))  
    num = int((size - 1) / 2)  
    input_image = np.pad(input_image, (num, num), mode="constant", constant_values=0) 
    m, n = input_image.shape  
    output_image = np.copy(input_image)  
    for i in range(num, m - num):
        for j in range(num, n - num):
            output_image[i, j] = np.sum(filter_template * input_image[i - num:i + num + 1, j - num:j + num + 1]) / (size ** 2)
    output_image = output_image[num:m - num, num:n - num]  
    return output_image

def zhongzhi(image, k = 3):
    imarray = np.copy(image)
    height, width = imarray.shape
    edge = int((k-1)/2)
    if height - 1 - edge <= edge or width - 1 - edge <= edge:
        print("The parameter k is to large.")
        return None
    new_arr = np.zeros((height, width), dtype = "uint8")
    for i in range(height):
        for j in range(width):
            if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
                new_arr[i, j] = imarray[i, j]
            else:
                new_arr[i, j] = np.median(imarray[i - edge:i + edge + 1, j - edge:j + edge + 1])
    new_im = Image.fromarray(new_arr)
    return new_im

#homework1 (1)--------------------------------------------------------------------------------
#img = getimg()
#img2 = rgb_histeq1(img)

#homework1 (2)--------------------------------------------------------------------------------
cvimg = cv2.imread("111.jpg")
'''imghsi = RGB2HSI(cvimg)
imarrhsi = np.array(imghsi)
i = imarrhsi[...,2]
i_res, i_hist = histeq(i)
imghsi[...,2] = i_res
imgrgb2 = HSI2RGB(imghsi)'''

#homework2 (1)----------------------------------------------------------------------------------
#imgjiao = jiaoyan_noise(cvimg,0.3)
#imggas = gas_noise(cvimg)

#homework2 (2)------------------------------------------------------------------------------------
#imgjiaojun = junzhi_filter(imgjiao, 3)
#imggasjun = junzhi_filter(imggas, 3)
#Rjiaojun, Gjiaojun, Bjiaojun = cv2.split(imgjiao)
#Rjiaojun = junzhi(Rjiaojun, 3)
#Gjiaojun = junzhi(Gjiaojun, 3)
#Bjiaojun = junzhi(Bjiaojun, 3)
#imgjiaojun = np.hstack((Rjiaojun, Gjiaojun, Bjiaojun))
#imgjiaojun = imgjiao.copy()
#imgjiaojun[:,:,0] = Bjiaojun*255
#imgjiaojun[:,:,1] = Gjiaojun*255
#imgjiaojun[:,:,2] = Rjiaojun*255
#imgjiaojun = np.append(Rjiaojun, Gjiaojun, Bjiaojun, axis = 0)


#homework2 (3)------------------------------------------------------------------------------------
'''Rgaszhong, Ggaszhong, Bgaszhong = cv2.split(imggas)
Rgaszhong = zhongzhi(Rgaszhong, 3)
Ggaszhong = zhongzhi(Ggaszhong, 3)
Bgaszhong = zhongzhi(Bgaszhong, 3)'''
#imgjiaojun = np.hstack((Rjiaojun, Gjiaojun, Bjiaojun))
'''imggaszhong = imggas.copy()
imggaszhong[:,:,0] = Bgaszhong
imggaszhong[:,:,1] = Ggaszhong
imggaszhong[:,:,2] = Rgaszhong'''
#imgjiaojun = np.append(Rjiaojun, Gjiaojun, Bjiaojun, axis = 0)

#print(np.shape(imgjiao))
#print(np.shape(imgjiaojun))


#showimg(img2)
#showimg(imgrgb2)
#showimg(imgjiao)
#showimg(imggas)
#showimg(imgjiaojun)
#showimg(imggasjun)
image_hist(cvimg)