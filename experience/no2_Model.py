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
if __name__ == "__main__":
    # reference：keras中文手册: https://keras.io/zh/models/model/    
    # 模型结构
    model = build_model()
    # 配置器
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0002), metrics=['accuracy'])
    # 模型打印    
    print(model.summary())
    pass