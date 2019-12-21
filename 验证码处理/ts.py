import os
import cv2 as cv
import numpy as np
import pandas as pd


#将图片二值化
def custom_threshold(image):
    '''
    image:像素矩阵
    '''
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #把输入图像灰度化
    h, w =gray.shape[:2]
    m = np.reshape(gray, [1,w*h])
    mean = m.sum()/(w*h)
    ret, binary =  cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    binary = cv.medianBlur(binary,5)
    return binary

#计算出一张图片中字母的区域
def chutupan(img, index):
    '''
    img:像素矩阵
    index:图片中第几个字母
    '''
    img_x = custom_threshold(img)
    contours, hierarchy = cv.findContours(img_x,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0
    for i in range(1, len(contours)):
        x1, y1, w1, h1 = cv.boundingRect(contours[i])
        if w1>w and h1>h:
            x, y, w, h = x1, y1, w1, h1
    return [x+index*30,y,w,h]
    

#计算出所有图片
def xibiao(path):
    '''
    path:需要框出的总图片路径
    '''
    img_names =os.listdir(path)
    dicz = dict()
    for name in img_names:
        img = cv.imread(path+name)
        dic = dict()
        for j in range(5):
            img_x = img[:,30*j:30*j+30]
            xibiao = chutupan(img_x, j)
            dic[j] = xibiao
        dicz[name] = dic
    return dicz

#将每张图片的字母框出
def kuang(path, save_path, xbs):
    '''
    path:需要框出的总图片路径
    save_path:保存需保存的文件夹路径
    xbs:字母的区域信息
    '''
    
    if os.path.exists(path) == False:
        os.makedirs(path)
    
    for i in xbs.keys():
        orign = cv.imread(path+i)
        for j in xbs[i].keys():
            x, y, w, h = xbs[i][j]
            orign = cv.rectangle(orign,(x,y),(x+w,y+h),(0,0,0),2)
        cv.imwrite(save_path+i, orign)
        

#将图片的区域信息写入csv
def writecsv(xbs, name):
    '''
     xbs:字母的区域信息
     name:需要保存为的文件名
    '''
    zong = list()
    for i in xbs.keys():
        lie = list()
        lie.append(i)
        for j in xbs[i].keys():
            lie = lie + xbs[i][j]
        zong.append(lie)
    zong = pd.DataFrame(zong)
    zong.to_csv(name,header=False,index=False)

#将框好的小图片组合成大图
def datu(path, now_tu):
    '''
    path:框好的小图片的路径
    '''
    paths = os.listdir(path)
    img_da = np.zeros((750,600,3)).astype(np.uint8)
    for i in range(len(paths)):
        img = cv.imread(path+paths[i])
        img_da[30*(i//4):30*(i//4)+30, 150*(i%4):150*(i%4)+150] = img
        
    cv.imwrite(now_tu, img_da)
    
path = 'img/'
save_path = 'data/'
xbs = xibiao(path)
kuang(path, save_path, xbs)
datu(save_path, 'da_tu.jpg')
writecsv(xbs,'data.csv')