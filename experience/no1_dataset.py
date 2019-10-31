import numpy as np
import pandas as pd
import math

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

TEST_MANIFEST_DIR = "./data/test_data.csv"
def ts_gen(path=TEST_MANIFEST_DIR,batch_size = Batch_size):
    data_list = pd.read_csv(path)
    data_list = np.array(data_list)[:Lens]
    print("Found %s train items."%len(data_list))
    print("list 1 is",data_list[0,-1])
    steps = math.ceil(len(data_list) / batch_size)    # 确定每轮有多少个batch

    while True:
        for i in range(steps):
            batch_list = data_list[i * batch_size : i * batch_size + batch_size]
            batch_x = np.array([file for file in batch_list[:,1:]])
            yield batch_x

import matplotlib.pyplot as plt
if __name__ == "__main__":
    path = "../data/train.csv"
    # pandas 有 read_csv、shape 和 head
    data = pd.read_csv(path)
    #profile = data.profile_report(title='Dataset')
    #profile.to_file(output_file='result/Report.html')
    #print("data.shape", data.shape)
    #print("data.head", data.head)
    print(data)

    # pd 转 numpy 自动去除 head
    data = np.array(data)
    data = data[:Lens] # 取 0 到 Lens-1 行数据
    print("Found %s train items."% len(data))
    print(data)
    print(data.shape)
    print(data[0,-1]) # 数据[0,max]位置的值
    print(len(data))
    print(len(data[0]))
    #print(data[1,:])    # 第一行[0,:]、第二行[1,:]、第三行[2,:]、...
    #print(data[:,-1])   # 最后一列 —— 标签
    
    # ——————————————————————正式调用子程序
    count = 1
    while count <= 3: 
        show_iter = xs_gen()
        for x,y in show_iter:
            x1 = x[0]
            y1 = y[0]
            break
        print(y)
        print(x1.shape)
        plt.plot(x1)
        plt.show()
        count = count + 1
    pass