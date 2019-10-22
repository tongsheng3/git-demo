import cv2
import numpy as np

weizhe = list() #申明全局变量用于保存鼠标获取x，y

#定义鼠标事件
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        weizhe.append([y,x])
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)
        
img = cv2.imread("saber.jpeg", cv2.IMREAD_GRAYSCALE) #读入图片 
cv2.namedWindow("image")       #创建窗口
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN) #绑定鼠标事件
cv2.imshow("image", img)      #展示图片

cv2.waitKey(0)    
cv2.destroyAllWindows()     #清空窗口


img = cv2.imread("saber.jpeg", cv2.IMREAD_GRAYSCALE)
U = np.mat(weizhe)

#计算仿射矩阵
X = np.mat([[25,24,1],[25,76,1],[75,50,1]])
M = np.array((X.T*X)**-1*X.T*U)

#生成每个像素点坐标
inx = np.argwhere(img)
#生成画布
nowimg = np.ones(101*101).reshape(101,101)*255
inx = np.argwhere(nowimg)

inxx = np.insert(inx, 2, values=np.ones(len(inx)), axis=1)
#进行仿射变换
nowinx = np.array(np.dot(inxx, M)).astype(np.int64)
zon = np.concatenate((nowinx,inx),axis=1)
zon = zon[(zon[:,0]>=0) & (zon[:,1]>=0)]


#对应像素点赋值
nowimg[zon[:,-2],zon[:,-1]] = img[zon[:,0],zon[:,1]]
nowimg = nowimg.astype(np.uint8)

cv2.imshow("image", nowimg)      #展示图片
cv2.imwrite('test.png', nowimg)    #保存图片

cv2.waitKey(0)    
cv2.destroyAllWindows()     #清空窗口
