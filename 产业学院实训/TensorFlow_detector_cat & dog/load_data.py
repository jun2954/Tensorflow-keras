import numpy as np
import tensorflow as tf
# 加载数据
traindata = np.load('data/traindata.npy')
trainlabel = np.load('data/trianlabel.npy')
testdata = np.load('data/testdata.npy')
testlabel = np.load('data/testlabel.npy')

# 数据预处理  [0 - 1 : 255.0] [-1 ~ 1 : 128.0]
traindata = (traindata - 128.0) / 128.0
testdata = (testdata - 128.0) / 128.0

# 转换one hot编码  1 : [0,1] 0: [1,0]
trainlabel_onehot = tf.keras.utils.to_categorical(
    y=trainlabel,num_classes=2
)
testlabel_onehot = tf.keras.utils.to_categorical(
    y=testlabel,num_classes=2
)
# 打乱数据   [b,w,h,c]
permutation = np.random.permutation(traindata.shape[0])
traindata = traindata[permutation,:]
trainlabel_onehot = trainlabel_onehot[permutation]

permutation = np.random.permutation(testdata.shape[0])
testdata = testdata[permutation,:]
testlabel_onehot = testlabel_onehot[permutation]

# 保存处理好数据
np.save('data/traindata_normal.npy',traindata)
np.save('data/trainlabel_onehot.npy',trainlabel_onehot)
np.save('data/testdata_normal.npy',testdata)
np.save('data/testlabel_onehot.npy',testlabel_onehot)













