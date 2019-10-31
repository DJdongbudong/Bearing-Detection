import numpy as np
import pandas as pd
import math

# step 1/3 数据生成器
#把标签转成oneHot
def convert2oneHot(index,Lens):
    hot = np.zeros((Lens,))
    hot[int(index)] = 1
    return(hot)
MANIFEST_DIR = "../data/train.csv"
Batch_size = 20
Lens = 640 # 取640为训练和验证截点。
# 训练样本生成器——然后使用 keras 的 fit_generator 就可以不断调用 yield 的返回值
def xs_gen(path=MANIFEST_DIR, batch_size=Batch_size, train=True, Lens=Lens):
    data_list = pd.read_csv(path)
    if train:
        data_list = np.array(data_list)[:Lens]            # 取前Lens行的训练数据
        print("Found %s train items."%len(data_list))
        print("list 1 is",data_list[0,-1])
        steps = math.ceil(len(data_list) / batch_size)    # 确定每轮有多少个batch
    else:
        data_list = np.array(data_list)[Lens:]            # 取Lens行后的验证数据
        print("Found %s test items."%len(data_list))
        print("list 1 is",data_list[0,-1])
        steps = math.ceil(len(data_list) / batch_size)    # 确定每轮有多少个batch
    while True:
        for i in range(steps):
            batch_list = data_list[i * batch_size : i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:,1:-1]])
            batch_y = np.array([convert2oneHot(label,10) for label in batch_list[:,-1]])
            yield batch_x, batch_y
# step 2/3 模型制造
import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
TIME_PERIODS = 6000 # 数据长度为6000个点
def build_model(input_shape=(TIME_PERIODS,),num_classes=10):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))
    # keras.layers.Conv1D(filters, kernel_size)
    # 当使用该层作为模型第一层时，需要提供 input_shape 参数
    
    # input_shape：第一层卷积层——输入数据
    # 6000 -> 3000
    model.add(Conv1D(16, 8, strides=2, activation='relu', input_shape=(TIME_PERIODS,1)))
    
    # 3000 -> 1500
    model.add(Conv1D(16, 8, strides=2, activation='relu', padding="same"))
    # 1500 -> 750
    model.add(MaxPooling1D(2))
    
    # 750 -> 375
    model.add(Conv1D(64, 4, strides=2, activation='relu',padding="same"))
    # 375 -> 188
    model.add(Conv1D(64, 4, strides=2, activation='relu',padding="same"))
    # 188 -> 94
    model.add(MaxPooling1D(2))

    # 94 -> 47
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    # 47 -> 24
    model.add(Conv1D(256, 4,strides=2, activation='relu',padding="same"))
    # 24 -> 12
    model.add(MaxPooling1D(2))

    # 12 -> 12 (strides=1,则输入=输出)
    model.add(Conv1D(512, 2, strides=1, activation='relu', padding="same"))
    # 12 -> 12
    model.add(Conv1D(512, 2, strides=1, activation='relu', padding="same"))
    # 12 -> 6
    model.add(MaxPooling1D(2))

    """model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))"""
    # GlobalAveragePooling1D：对于时序数据的全局平均池化。
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    
    # Dense()：全连接层——返回计算类别
    model.add(Dense(num_classes, activation='softmax'))
    return model

# step 3/3 模型优化器和训练
if __name__ == "__main__":
    # 1 数据初始化
    train_iter = xs_gen()
    val_iter = xs_gen(train=False)

    # 2.模型保存点
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.4f}.h5',
        monitor='val_loss', save_best_only=True,verbose=1)

    # 3.模型构建
    model = build_model()

    # 4.损失函数与优化器
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])
    # 模型打印：# print(model.summary())
    
    Long = 792
    # 5.模型训练，配合使用数据生成器
    model.fit_generator(
        generator = train_iter,
        steps_per_epoch = Lens//Batch_size,
        epochs = 50,
        initial_epoch = 0,
        validation_data = val_iter,
        nb_val_samples = (Long - Lens)//Batch_size,
        callbacks = [ckpt],
        )
    # 6.训练后的模型保存
    model.save("finishModel.h5")
    pass
