import tensorflow as tf
import numpy as np
import cv2

# 预训练文件
model = tf.keras.models.load_model('./weight/newmodelnets.h5')
testdata = np.load('./data/testdata_normal.npy')
testlabel = np.load('./data/testlabel_onehot.npy')
# result = model.evaluate(testdata,testlabel)

def predict_img(path):
    images = cv2.imread(path)
    images = cv2.cvtColor(images,cv2.COLOR_BGR2RGB)
    images = cv2.resize(images,(224,224))
    images = (images - 128.0) / 128.0
    images = np.expand_dims(images,axis=0)  # 扩张维度
    return images

test1 = predict_img('./images/test1.jpg')
test2 = predict_img('./images/test2.jpg')

x = np.concatenate((test1,test2),axis=0)
y_predict = model.predict(x)
print(y_predict)

y = ['猫','狗']
print(y[np.argmax(y_predict[0])])
print(y[np.argmax(y_predict[1])])













