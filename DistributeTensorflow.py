'''
Created on 2017年5月28日

@author: weizhen
'''
import tensorflow as tf
c = tf.constant("Hello distributed TensorFlow!")
#创建一个本地Tensorflow集群
server = tf.train.Server.create_local_server()
#在集群上创建一个会话
sess = tf.Session(server.target)
#输出Hello,distribute tensorflow
print(sess.run(c))

#首先通过tf.train.Server.create_local_server函数在本地建立了一个只有一台机器的Tensorflow集群
#然后在该集群上生成了一个会话，并通过生成的会话将运算运行在创建的Tensorflow集群上。
#虽然这只是一个单机集群，但它大致反映了TensorFlow集群的工作流程。
#Tensorflow集群通过一系列的任务来执行TensorFlow计算图中的运算
#一般来说不同任务跑在不同机器上。
#最主要的例外是使用GPU时，不同任务可以使用同一台机器上的不同GPU
#Tensorflow集群中的任务也会被集合成工作
#每个工作可以包含一个或者多个任务。
#在训练深度学习模型时，一台运行反向传播的机器是一个任务，而所有运行反向传播机器的集合是一种工作