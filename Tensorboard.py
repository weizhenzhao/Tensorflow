'''
Created on 2017年5月22日

@author: weizhen
'''
import tensorflow as tf
#定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1,input2], name="add")

#生成一个写日志的writer,并将当前的Tensorflow计算图写入日志。TensorFlow提供了多种写日志文件的API
writer=tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
writer.close()
#Tensorboard会自动读取最新的Tensorflow日志文件，并呈现当前Tensorflow程序运行的最新状态
#在TensorFlow安装完成时，TensorBoard会被自动安装。运行下面的命令可以启动TensorBoard
#运行TensorBoard,并将日志的地址指向上面程序日志输出的地址
#tensorboard --logdir=/path/to/log
