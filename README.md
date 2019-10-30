# Bearing Detection
 bearing detection by conv1d
# 数据介绍
    轴承有3种故障：外圈故障，内圈故障，滚珠故障，外加正常的工作状态。如表1所示，结合轴承的3种直径（直径1,直径2,直径3），轴承的工作状态有10类：
![image](https://github.com/DJdongbudong/Bearing-Detection/blob/master/resource/1.png)
参赛选手需要设计模型根据轴承运行中的振动信号对轴承的工作状态进行分类。

1.train.csv，训练集数据，1到6000为按时间序列连续采样的振动信号数值，每行数据是一个样本，共792条数据，第一列id字段为样本编号，最后一列label字段为标签数据，即轴承的工作状态，用数字0到9表示。

2.test_data.csv，测试集数据，共528条数据，除无label字段外，其他字段同训练集。 总的来说，每行数据除去id和label后是轴承一段时间的振动信号数据，选手需要用这些振动信号去判定轴承的工作状态label。
# 评分算法
采用各个品类F1指标的算术平均值，它是Precision 和 Recall 的调和平均数。
![image](https://github.com/DJdongbudong/Bearing-Detection/blob/master/resource/2.png)
其中，Pi是表示第i个种类对应的Precision， Ri是表示第i个种类对应Recall。

# 赛题分析
    10分类的问题，输入的形状为(-1,6000)，网络输出的结果为(-1,10)

# 实现过程
1. 数据读取与处理
2. 网络模型搭建
3. 模型的训练
4. 模型应用与提交预测结果
