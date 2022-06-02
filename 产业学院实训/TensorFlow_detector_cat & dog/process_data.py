import glob
import os
import numpy as np
import cv2
import tqdm  # 文件
from tqdm import tqdm
def load_data(path,label):
    img_path = glob.glob(os.path.join(path,'*.jpg'))
    print(img_path)
    # 创建空的多维数组  #  [w,h,c] ->  [b,w,h,c]    # 默认float64  ->  float32
    x = np.empty([len(img_path),224,224,3],dtype=np.float32)   # vggnet16 或者 轻量级神经网络
    y = np.empty([0],dtype=np.float32)

    print(x.shape,y.shape)
    for index,img_data in enumerate(img_path):
        img_data = cv2.imread(img_data)
        img_data = cv2.cvtColor(img_data,cv2.COLOR_BGR2RGB)
        img_data = cv2.resize(img_data,(224,224))
        # cv2.imshow('im_data',img_data)
        # cv2.waitKey(0)
        x[index,:] = img_data
    y = np.linspace(label,label,x.shape[0])
    return x,y

train_catx , train_caty = load_data('images/training_set/cats',0)
train_dogx , train_dogy = load_data('images/training_set/dogs',1)
test_catx , test_caty = load_data('images/test_set/cats',0)
test_dogx , test_dogy = load_data('images/test_set/dogs',1)

# 获取训练集测试集 特征值与目标值
triandata = np.concatenate((train_catx,train_dogx),axis=0)
trianlabel = np.concatenate((train_caty,train_dogy),axis=0)
testdata = np.concatenate((test_catx,test_dogx),axis=0)
testlabel = np.concatenate((test_caty,test_dogy),axis=0)

# 测试根据索引下表 取值 与保存数据
trainimage = cv2.cvtColor(triandata[224].astype(np.uint8),cv2.COLOR_RGB2BGR)
testimage = cv2.cvtColor(testdata[1888].astype(np.uint8),cv2.COLOR_RGB2BGR)
cv2.imwrite('images/test1.jpg', trainimage)
cv2.imwrite('images/test2.jpg', testimage)
print((trianlabel[224],testlabel[1888]))

np.save('data/traindata.npy',triandata)
np.save('data/trianlabel.npy',trianlabel)
np.save('data/testdata.npy',testdata)
np.save('data/testlabel.npy',testlabel)
