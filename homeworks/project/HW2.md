# HW2 - RBF-NN

![Screen Shot](imgs/hw2.gif)

继续沿用了 Utopia 上周给出的基础设施，只基于 Eigen 库实现了简单的 RBF 神经网络。

每帧执行一次（Forward + Backward）迭代并可视化当前模型的预测值，可实时调整的参数：
* Fitting Step：拟合步长，指定隔几个像素取一个采样点
* Input Points：点击画布即可添加训练数据，改变此参数会重新开始训练
* Hidden Node Count：指定隐层的节点数，可以和输入样本数不同，改变此参数会重新开始训练
* Learning Rate：全局学习速率，手动指定
* Loss Function：切换损失函数，实现了 Mean Squared 和 Cross Entropy 两种

比较重要的一些优化：
* 输入数据统一从屏幕空间归一化，可以减小数值精度不足带来的很多问题
* 训练参数要合理初始化，在有效范围内小幅随机赋初值
