import numpy as np
import pandas as pd
import math

# step 1/2 数据生成器
Batch_size = 20
Lens = 528 # 取640为训练和验证截点。
TEST_MANIFEST_DIR = "../data/test_data.csv"

def ts_gen(path = TEST_MANIFEST_DIR, batch_size = Batch_size):
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

from keras.models import *
import matplotlib.pyplot as plt
# step 2/2 模型优化器和训练
if __name__ == "__main__":
    # 测试数据的模型检测
    test_iter = ts_gen()
    model = load_model("best_model.40-0.0011.h5")
    pres = model.predict_generator(
        generator=test_iter,
        steps=math.ceil(528/Batch_size),
        verbose=1
        )
    print(pres.shape)
    ohpres = np.argmax(pres,axis=1)
    print(ohpres.shape)
    
    # 数据写入文件
    df = pd.DataFrame()
    df["id"] = np.arange(1,len(ohpres)+1)
    df["label"] = ohpres
    df.to_csv("submmit.csv",index=None)
    
    # 其他（可略）
    test_iter = ts_gen()
    for x in test_iter:
        x1 = x[0]
        break
    plt.plot(x1)
    plt.show()
    pass
