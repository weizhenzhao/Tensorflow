'''
Created on 2017年5月21日

@author: weizhen
'''
#Tensorflow的另外一个高层封装TFLearn(集成在tf.contrib.learn里)对训练Tensorflow模型进行了一些封装
#使其更便于使用。
#使用TFLearn实现分类问题
#为了方便数据处理，本程序使用了sklearn工具包，
#更多信息可以参考http://scikit-learn.org
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
#导入TFLearn
learn = tf.contrib.learn

#自定义模型，对于给定的输入数据(features)以及其对应的正确答案(target)
#返回在这些输入上的预测值、损失值以及训练步骤
def my_model(features,target):
    #将预测的目标转换为one-hot编码的形式，因为共有三个类别，所以向量长度为3.经过转化后，第一个类别表示为(1,0,0)
    #第二个为(0,1,0)，第三个为(0,0,1)
    target = tf.one_hot(target,3,1,0)
    
    #定义模型以及其在给定数据上的损失函数
    logits = tf.contrib.layers.fully_connected(features,3,tf.nn.softmax)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    
    #创建模型的优化器，并得到优化步骤
    train_op=tf.contrib.layers.optimize_loss(loss,          #损失函数
                                             tf.contrib.framework.get_global_step(), #获取训练步数并在训练时更新
                                             optimizer='Adam',   #定义优化器
                                             learning_rate=0.01)     #定义学习率
    #返回在给定数据上的预测结果、损失值以及优化步骤
    return tf.arg_max(logits, 1),loss,train_op

#加载iris数据集，并划分为训练集合和测试集合
iris = datasets.load_iris()
x_train,x_test,y_train,y_test=model_selection.train_test_split(iris.data,
                                                                iris.target,
                                                                test_size=0.2,
                                                                random_state=0)
#将数据转化为float32格式
x_train,x_test = map(np.float32,[x_train,x_test])
#封装和训练模型，输出准确率
classifier=SKCompat(learn.Estimator(model_fn=my_model,model_dir="Models/model_1"))
#使用封装好的模型和训练数据执行100轮迭代
classifier.fit(x_train,y_train,steps=800)

#使用训练好的模型进行结果预测
y_predicted=[i for i in classifier.predict(x_test)]
#计算模型的准确度
score=metrics.accuracy_score(y_test,y_predicted)
print("Accuracy: %.2f"%(score*100))


