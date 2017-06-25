'''
Created on 2017年5月28日

@author: weizhen
'''
#当一个Tensorflow集群有多个任务时，需要使用
#tf.train.ClusterSpec来指定运行每一个任务的机器。
#以下代码展示了在本地运行有两个任务的Tensorflow集群
import tensorflow as tf
c=tf.constant("Hello from server!")

#生成一个有两个任务的集群，一个任务跑在本地2222端口，另外一个跑在本地2223端口
cluster = tf.train.ClusterSpec({"local":["localhost:2222","localhost:2223"]})
#通过上面生成的集群配置生成Server,并通过job_name和task_index指定当前所启动的任务
#因为该任务是第一个任务，所以task_index为0
server=tf.train.Server(cluster,job_name="local",task_index=0)

#通过server.target生成会话来使用Tensorflow集群中的资源。通过设置
#log_device_placement可以看到执行每一个操作的任务
sess = tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
server.join()