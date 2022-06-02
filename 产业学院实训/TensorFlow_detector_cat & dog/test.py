import matplotlib.pyplot as plt
import cv2
import numpy as np

img1  = cv2.imread('./images/training_set/cats/cat.1.jpg')
img1 = cv2.resize(img1,(224,224))
img2  = cv2.imread('./images/training_set/dogs/dog.1.jpg')
img2 = cv2.resize(img2,(224,224))

x = np.concatenate((np.array(img1),np.array(img2)),axis=0)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.waitKey(0)
plt.imshow(x)
plt.show()