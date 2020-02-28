from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random

#获取图片
def getimg():
  return Image.open("1.jpg")
  
#显示图片
def showimg(img, isgray=False):
  plt.axis("off")
  if isgray == True:
    plt.imshow(img, cmap='gray')
  else: 
    plt.imshow(img)
  plt.show()

def GausLow(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform(d):
        trans = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(trans.shape[0]):
            for j in range(trans.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                trans[i,j] = np.exp(-(dis**2)/(2*(d**2)))
        return trans
    d = make_transform(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d)))
    return new_img, d

def GausHigh(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform(d):
        trans = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(trans.shape[0]):
            for j in range(trans.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                trans[i,j] = 1-np.exp(-(dis**2)/(2*(d**2)))
        return trans
    d = make_transform(d)
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d)))
    return new_img, d

def dct(img):
    #print(img.shape)
    img_dct = np.zeros(img.shape,dtype=np.float)
    h = img.shape[0]
    w = img.shape[1]
    num_h=int(np.floor(h/8))
    num_w=int(np.floor(w/8))
    num=0
    for i in range(num_h):
        for j in range(num_w):
            num+=1
            sub=img[8*i:8*(i+1),8*j:8*(j+1)]
            sub1=sub.astype('float')
            C_temp=np.zeros(sub.shape)
            dst=np.zeros(sub.shape)                        
            m, n=sub.shape
            N=n
            C_temp[0, :]=1*np.sqrt(1/N)             
            for i in range(1, m):
                 for j in range(n):
                      C_temp[i, j]=np.cos(np.pi*i*(2*j+1)/(2*N))*np.sqrt(2/N)             
            dst=np.dot(C_temp, sub1)
            dst=np.dot(dst, np.transpose(C_temp))             
            dst1= np.log(abs(dst))  
            #showimg(sub1) 
            #showimg(dst1)  
            #print(dst1.shape[0])
            img_new = np.dot(np.transpose(C_temp) , dst)
            img_new = np.dot(img_new, C_temp)


            # img[]
            img[8*i:8*(i+1),8*j:8*(j+1)]=img_new
            img_dct[8*i:8*(i+1),8*j:8*(j+1)]= dst1
            #np.copy(img_dct[8*i:8*(i+1),8*j:8*(j+1)],np.array(dst1).all)
            '''
            for key1 in range(8):
                for key2 in range(8):
                    img[8*i+key1,8*j+key2]=img_new[key1][key2]
                    #img_dct[8*i+key1,8*j+key2]=dst1[key1][key2]
                    np.copy(img[8*i+key1,8*j+key2], img_new[key1][key2])
            '''
            #showimg(img_dct[8*i:8*(i+1),8*j:8*(j+1)]-dst1)
            #print(img_dct[8*i:8*(i+1),8*j:8*(j+1)].shape)
            #print(dst1.shape)
            #showimg(img_dct[8*i:8*(i+1),8*j:8*(j+1)])
            #showimg(dst1)

    return img, img_dct



#img = getimg()
img = cv2.imread("3.jpg")
Rimg, Gimg, Bimg = cv2.split(img)


#homework1--------------------------------------------------------------------------------------------

#question1--------------------------------------------------------------------------------------------

'''Rf = np.fft.fft2(Rimg)
Rfshift = np.fft.fftshift(Rf)
s1 = np.log(np.abs(Rfshift))
Rgauslspace, Rgauslfreq = GausLow(Rimg, 250)

Gf = np.fft.fft2(Gimg)
Gfshift = np.fft.fftshift(Gf)
s1 = np.log(np.abs(Gfshift))
Ggauslspace, Ggauslfreq = GausLow(Gimg, 250)

Bf = np.fft.fft2(Bimg)
Bfshift = np.fft.fftshift(Bf)
s1 = np.log(np.abs(Bfshift))
Bgauslspace, Bgauslfreq = GausLow(Bimg, 250)

gauslspace = img.copy()
gauslspace[:,:,0] = Rgauslspace
gauslspace[:,:,1] = Ggauslspace
gauslspace[:,:,2] = Bgauslspace

showimg(gauslspace)
showimg(Rgauslfreq)
showimg(Ggauslfreq)
showimg(Bgauslfreq)'''

#question2--------------------------------------------------------------------------------------------

Rf = np.fft.fft2(Rimg)
Rfshift = np.fft.fftshift(Rf)
s1 = np.log(np.abs(Rfshift))
Rgauslspace, Rgauslfreq = GausHigh(Rimg, 150)

Gf = np.fft.fft2(Gimg)
Gfshift = np.fft.fftshift(Gf)
s1 = np.log(np.abs(Gfshift))
Ggauslspace, Ggauslfreq = GausHigh(Gimg, 150)

Bf = np.fft.fft2(Bimg)
Bfshift = np.fft.fftshift(Bf)
s1 = np.log(np.abs(Bfshift))
Bgauslspace, Bgauslfreq = GausHigh(Bimg, 150)

gauslspace = img.copy()
gauslspace[:,:,0] = Rgauslspace
gauslspace[:,:,1] = Ggauslspace
gauslspace[:,:,2] = Bgauslspace

showimg(gauslspace)
showimg(Rgauslfreq)
showimg(Ggauslfreq)
showimg(Bgauslfreq)

#homework2--------------------------------------------------------------------------------------------

'''Rnew, Rdct = dct(Rimg)
Gnew, Gdct = dct(Gimg)
Bnew, Bdct = dct(Bimg)

DCTIMG = img.copy()
DCTIMG[:,:,0] = Rdct
DCTIMG[:,:,1] = Gdct
DCTIMG[:,:,2] = Bdct

IDCTIMG = img.copy()
IDCTIMG[:,:,0] = Rnew
IDCTIMG[:,:,1] = Gnew
IDCTIMG[:,:,2] = Bnew'''

'''Rdct = Rimg.astype('float')
Gdct = Gimg.astype('float')
Bdct = Bimg.astype('float')

Rdct = cv2.dct(Rdct)
Gdct = cv2.dct(Gdct)
Bdct = cv2.dct(Bdct)

DCTIMG = img.copy()
DCTIMG[:,:,0] = Rdct
DCTIMG[:,:,1] = Gdct
DCTIMG[:,:,2] = Bdct'''

#showimg(DCTIMG)
#showimg(IDCTIMG)



