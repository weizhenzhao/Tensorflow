'''
Created on 2017年5月28日

@author: weizhen
'''
#下面给出第二个任务的代码
import tensorflow as tf
c = tf.constant("Hello from server2!")
#和第一个程序一样集群配置。集群中的每一个任务需要采用相同的配置
cluster = tf.train.ClusterSpec({"local":["localhost:2222","localhost:2223"]})
#指定task_index为1,所以这个程序将在localhost:2223启动服务
server = tf.train.Server(cluster,job_name="local",task_index=1)
#通过server.target生成会话来使用Tensorflow集群中的资源。通过设置
#log_device_placement可以看到执行每一个操作的任务
sess = tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
server.join()
#当只启动第一个任务是，程序会停下来等待第二个任务启动，而且持续输出failed to connect to 'ipv4:127.0.0.1:2223'
#当第二个任务启动后，可以看到从第一个任务中会输出Hello from server1的结果

#和使用多GPU类似，Tensorflow支持通过tf.device来指定操作运行在哪个任务上
#如果将第二个任务中定义计算的语句改为以下代码，就可以看到这个计算将被调度到cpu:0上面
#with tf.device("/job:local/task:1"):
#   c = tf.constant("Hello from server2!")

#一般在训练神经学习模型时，会定义两个工作。一个工作专门负责存储、获取以及更新变量的取值
#这个工作所包含的任务统称为参数服务器。
#另外一个工作负责运行反向传播算法来获取参数梯度，这个工作所包含的任务统称为计算服务器
#
#一个常见的用于训练深度学习的模型的Tensorflow集群配置方法
#tf.train.ClusterSpec({
#    "worker":[
#        "tf-worker0:2222",
#        "tf-worker1:2222",
#        "tf-worker2:2222"
#        ],
#        "ps":[
#        "tf-ps0:2222",
#        "tf-ps1:2222"
#    ]})

#使用分布式Tensorflow训练深度学习模型一般有两种方式。一种方式叫做计算图内分布式
#使用这种分布式训练方式时，所以的任务都会使用一个TensorFlow计算图中的变量
#而只是将计算部分发布到不同的计算服务器上
#然而因为计算图内分布式需要有一个中心节点来生成这个计算图并分配计算任务
#所以当数据量太大时，这个中心节点容易造成性能瓶颈


#另外一种分布式Tensorflow训练深度学习模型的方式叫计算图之间分布式。
#使用这种分布式方式时，在每一个计算服务器上都会创建一个独立的TensorFlow计算图，
#但不同计算图中的相同参数需要以一种固定的方式放到同一个参数服务器上
#TensorFlow提供了tf.train.replica_device_setter函数来帮助完成这一个过程
#因为每个计算服务器的TensorFlow计算图是独立的
#所以这种方式的并行度要更高。但在计算图之间分布式下进行参数的同步更新比较困难
#为了解决这个问题，TensorFlow提供了tf.train.SyncReplicasOptimizer函数来了帮助实现参数的同步更新
#这让计算图之间分布式方式被更加广泛地使用









