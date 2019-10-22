import cv2
import numpy as np

weizhe = list()
#鼠标事件
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        weizhe.append([y,x,1])
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


#获取第一张图的点
img = cv2.imread('klcc_a.png', cv2.IMREAD_GRAYSCALE)
weizhe = list()
cv2.namedWindow('image')
loc = cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
weizhe1 = np.mat(weizhe)

#获取第一张图的点
img = cv2.imread('klcc_b.png', cv2.IMREAD_GRAYSCALE)
weizhe = list()
cv2.namedWindow('image')
loc = cv2.setMouseCallback('image', on_EVENT_LBUTTONDOWN)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
weizhe2 = np.array(weizhe)



#计算仿射矩阵
w = np.dot((weizhe1.T*weizhe1)**-1*weizhe1.T, weizhe2[:,:-1])
A = np.insert(np.array(w), 2, values=[0,0,1], axis=1)
#计算其逆矩阵
B = np.mat(A)**-1

#重新加载图片
imga = cv2.imread('klcc_a.png', cv2.IMREAD_GRAYSCALE)
imgb = cv2.imread('klcc_b.png', cv2.IMREAD_GRAYSCALE)

#获取得每个点的下标
index = np.argwhere(imgb>=0)
index = np.insert(index, 2, values=np.ones(len(index),np.int64), axis=1)
#获取出去新图片需要的尺寸
bh = np.array(np.dot(index, B[:,:-1]))
Xmax = (max(bh[:,0]) if max(bh[:,0])>imga.shape[0] else imga.shape[0])
Xmin = min(bh[:,0])
if Xmin>0: 
    Xmin = 0
Ymax = (max(bh[:,1]) if max(bh[:,1])>imga.shape[1] else imga.shape[1])
Ymin = min(bh[:,1])
if Ymin>0:
    Ymin = 0

#创建新图片    
M = int(Xmax - Xmin + 1)
N = int(Ymax - Ymin + 1)
imgc = np.ones(M*N, dtype=np.int).reshape(M,N)


ind3 = np.argwhere(imgc)
#换算参考坐标系
inc = ind3 + np.array([int(Xmin-1),int(Ymin-1)])

inc = np.concatenate((inc,ind3),axis=1) #合并二维数组4

Ma,Na = imga.shape
Mb,Nb = imgb.shape

#筛选出有效坐标系
yuantu = inc[(inc[:,0]>=0) & (inc[:,0]<Ma) & (inc[:,1]>=0) & (inc[:,1]<Na)]
#筛选出无效效坐标系
pingjie = inc[(inc[:,0]<0) | (inc[:,0]>Ma) | (inc[:,1]<0) | (inc[:,1]>Na)]

#将无效坐标系进行处理
chuli = np.insert(pingjie[:,:2],2,values=np.ones(len(pingjie)),axis=1)
pingjie[:,:2] = np.round(np.dot(chuli, A[:,:-1]), decimals=0).astype(np.int32)

pingjie = pingjie[(pingjie[:,0]>0) & (pingjie[:,0]<Mb) & (pingjie[:,1]>0) & (pingjie[:,1]<Nb)]

#对应位置像素赋值
imgc[yuantu[:,2],yuantu[:,3]] = imga[yuantu[:,0],yuantu[:,1]]
imgc[pingjie[:,2],pingjie[:,3]] = imgb[pingjie[:,0],pingjie[:,1]]


imgc.astype(np.uint8) #展示图像需要的格式

#保存图片
cv2.imwrite('ce.png', imgc)