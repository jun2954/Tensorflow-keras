import tensorflow as tf
import numpy as np
# 轻量级神经网络    pip install pydot文档
# 自定义神经网络  （从经典神经网络里面提取一部分，自己定义一部分）
modelnets = tf.keras.applications.MobileNet(input_shape=(224,224,3), # 输入层
                                weights='imagenet')
tf.keras.utils.plot_model(modelnets,show_shapes=True,to_file='./images/model.png')
x = modelnets.get_layer('conv_pw_13_relu').output
x = tf.keras.layers.Flatten()(x)
x =  tf.keras.layers.Dense(units=100,activation='relu')(x)  # 隐藏层
predictlayer = tf.keras.layers.Dense(units=2,activation='softmax')(x)   # 输出层
# 创建自定义网络
newmodelnets = tf.keras.Model(inputs=modelnets.inputs,outputs=predictlayer)
tf.keras.utils.plot_model(newmodelnets,show_shapes=True,to_file='./images/model1.png')

for layer in newmodelnets.layers:
    layer.trainable = False  # 查看每层是否参见训练
newmodelnets.layers[-1].trainable = True
newmodelnets.layers[-2].trainable = True
for layer in newmodelnets.layers:
    print(layer.trainable)  # 查看每层是否参见训练

# 交叉熵损失函数
# categorical_crossentropy : one_hot 编码  [0,1]  [1,0]
# sparse_categorical_crossentropy : 标签编码 [0,1,2,3,4...]
newmodelnets.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])
newmodelnets.summary()  # 收集以及总结概括网络
# 可视化
tensorboard = tf.keras.callbacks.TensorBoard(
    # log_dir:事件文件存储路径 ,write_images:是否可视化训练图片,histogram_freq:存储计算模型权重激活函数层直方图
    # write_graph : 是否写入数据流图
    log_dir='model',write_images=1,histogram_freq=1,write_graph=True
)
traindata = np.load('./data/traindata_normal.npy')
trainlabel = np.load('./data/trainlabel_onehot.npy')
# x : 特征值 ,y: 目标值  epochs：迭代次数 ，
# batch_size：每次迭代训练样本的大小[16,32,64,128,256]
# validation_split : 验证集大小 20%  verbose: 详情模式 0静音 1进度条 2每个迭代次数占一行
newmodelnets.fit(
# todo : tensorboard --logdir='./'
    x=traindata,y=trainlabel,epochs=3,batch_size=64,
    validation_split=0.2,callbacks=[tensorboard],verbose=1
)
newmodelnets.save('./weight/newmodelnets.h5')