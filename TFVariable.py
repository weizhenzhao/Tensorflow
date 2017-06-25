'''
Created on Apr 21, 2017

@author: P0079482
'''
#如何通过tf.variable_scope函数来控制tf.ger_variable函数获取已经创建过的变量
#在名字为foo的命名空间内创建名字为v的变量
import tensorflow as tf
with tf.variable_scope("foo"):
    v = tf.get_variable("v",shape=[1],initializer=tf.constant_initializer(1.0))
#因为在命名空间foo中已经存在名为v的变量，所有下面的代码将会报错:
#Variable foo/v already exists,
with tf.variable_scope("foo"):
    v = tf.get_variable("v",[1])
    
#在生成上下文管理器时，将参数reuse设置为True.这样tf.get_variable函数将直接获取已经声明的变量
with tf.variable_scope("foo",reuse=True):
    v1 = tf.get_variable("v",[1])
    print(v==v1) #输出为True,代表v,v1代表的是相同的Tensorflow中的变量
    
#将参数reuse设置为True是，tf.variable_scope将只能获取已经创建过的变量。
#因为在命名空间bar中还没有创建变量v,所以下面的代码将会报错
with tf.variable_scope("bar",reuse=True):
    v = tf.get_variable("v",[1])
        
#如果tf.variable_scope函数使用reuse=None或者reuse=False创建上下文管理器
#tf.get_variable操作将创建新的变量。
#如果同名的变量已经存在，则tf.get_variable函数将报错
#Tensorflow中tf.variable_scope函数是可以嵌套的
with tf.variable_scope("root"):
    #可以通过tf.get_variable_scope().reuse函数来获取上下文管理器中reuse参数的值
    print(tf.get_variable_scope().reuse) #输出False,即最外层reuse是False
    
    with tf.variable_scope("foo",reuse=True): #新建一个嵌套的上下文管理器并指定reuse为True
        print(tf.get_variable_scope().reuse)    #输出True
        with tf.variable_scope("bar"):        #新建一个嵌套的上下文管理器，但不指定reuse，这时reuse的取值会和外面一层保持一致
            print(tf.get_variable_scope().reuse)    #输出True
    print(tf.get_variable_scope().reuse)            #输出False

#tf.variable_scope函数生成的上下文管理器也会创建一个Tensorflow中的命名空间
#在命名空间内创建的变量名称都会带上这个命名空间作为前缀
#所以tf.variable_scope函数除了可以控制tf.get_variable执行的功能之外
#这个函数也提供了一个管理命名空间的方式
v1 = tf.get_variable("v",[1])
print(v1.name)#输出v:0  "v"为变量的名称，":0"表示这个变量是生成变量这个运算的第一个结果


with tf.variable_scope("foo"):
    v2 = tf.get_variable("v",[1])
    print(v2.name)#输出foo/v:0 在tf.variable_scope中创建的变量，名称前面会
                  #加入命名空间的名称，并通过/来分隔命名空间的名称和变量的名称

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v",[1])
        print(v3.name)  #输出foo/bar/v:0  命名空间可以嵌套，同时变量的名称也会加入所有命名空间的名称作为前缀
    
    v4 = tf.get_variable("v1",[1])
    print(v4.name) #输出foo/v1:0  当命名空间退出之后，变量名称也就不会再被加入其前缀了
    
#创建一个名称为空的命名空间，并设置reuse=True
with tf.variable_scope("",reuse=True):
    v5=tf.get_variable("foo/bar/v",[1])#可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量。
    
    print(v5==v3)
    v6=tf.get_variable("foo/v1",[1])
    print(v6==v4)

#通过tf.variable_scope和tf.get_variable函数，以下代码对inference函数的前向传播结果做了一些改进
def inference(input_tensor,reuse=False):
    #定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope('layer1',reuse=reuse):
        #根据传进来的reuse来判断是创建新变量还是使用已经创建好了。在第一次构造网络时需要创建新的变量，
        #以后每次调用这个函数都直接使用reuse=True就不需要每次将变量传进来了
        weights= tf.get_variable("weights",[INPUT_NODE,LAYER1_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases= tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    #类似地定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope('layer2',reuse=reuse):
        weights=tf.get_variable("weights",[LAYER1_NODE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases=tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
    #返回最后的前向传播结果
    return layer2

x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
y=inference(x)

#在程序中需要使用训练好的神经网络进行推倒时，可以直接调用inference(new_x,True)




#Tensorflow模型持久化
import tensorflow as tf
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Varibale(tf.constant(2.0,shape=[1]),name="v2")
result=v1+v2
init_op=tf.initialize_all_variables()
#声明tf.train.Saver类用于保存模型
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    #将模型保存到/path/to/model/model.ckpt文件
    saver.save(sess,"/path/to/model/model.ckpt")
#生成的文件
#model.ckpt.meta保存了TensorFlow计算图的结构
#model.ckpt这个文件保存了TensorFlow程序中每一个变量的取值
#checkpoint文件 保存了一个目录下所有的模型文件列表




#加载保存的文件
import tensorflow as tf
#使用和保存模型代码中一样的方式来声明变量
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0,shpae=[1]),name="v2")
result=v1+v2

saver=tf.train.Saver()
with tf.Session() as sess:
    #加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess,"/path/to/model/model.ckpt")
    print(sess.run(result))


import tensorflow as tf
#直接加载持久化的图
saver=tf.train.import_meta_graph("/path/to/model/model.ckpt/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,"/path/to/model/model.ckpt")
    #通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensorflow_by_name("add:0")))
    #输出[ 3.]
    
#加载指定的变量
#可能之前有一个训练好的五层神经网络模型，但现在想尝试一个六层的神经网络
#那么可以将前面五层神经网络中的参数直接加载到新的模型，而仅仅将最后一层神经网络重新训练
#在加载模型的代码中使用saver=tf.train.Saver([v1])命令来构建tf.train.Saver类
v1=tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")
#如果直接使用tf.train.Saver()来加载模型会报找不到的错误
#使用字典来重命名就可以加载原来的模型了，这个字典指定了原来名称为v1的变量现在加载到变量v1中（名称为other-v1）
#名称为v2的变量加载到变量v2中（名称为other-v2）
saver=tf.train.Saver({"v1":v1,"v2",v2})
#这样做的主要目的之一就是方便使用变量的滑动平均值



#给出了一个保存滑动平均模型的样例
import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,name="v")
#在没有申明滑动平均模型时只有一个变量v，所以下面的语句只会输出"v:0
for variables in tf.all_variables():
    print(variables.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op=ema.apply(tf.all_variables())
#在申明滑动平均模型之后，Tensorflow会自动生成一个影子变量
#v/ExponentialMoving Average 于是下面的语句会输出
#"v:0" 和 "v/ExponentialMovingAverage:0"
for variables in tf.all_variables():
    print(variables.name)


saver = tf.train.Saver()
with tf.Session() as sess:
    init_op=tf.initialize_all_variables()
    sess.run(init_op)
    
    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    #保存时Tensorflow会将v:0和v/ExponentialMovingAverage:0两个变量都存下来
    saver.save(sess,"/path/to/model/model.ckpt")
    print(sess.run([v,ema.average(v)]))  #输出[10.0，0.099999905]


#一下代码给出了如何通过变量重命名直接读取变量的滑动平均值
v=tf.Variable(0,dtype=tf.float32,name="v")
#通过变量重命名将原来变量v的滑动平均值直接赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"/path/to/model/model.ckpt")
    print(sess.run(v))
    

#为了方便加载时重命名滑动平均变量，tf.train.ExponentialMovingAverage类提供了
#variables_to_restore函数来生成tf.train.Saver类所需要的变量重命名字典
import tensorflow as tf
v=tf.Variable(0,dtype=tf.float32,name="v")
ema=tf.train.ExponentialMovingAverage(0.99)
#通过使用variables_to_restore函数可以直接生成上面代码中提供的字典
#{"v/ExponentialMovingAverage":v}
print(ema.variables_to_restore())
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess,"/path/to/model/model.ckpt")
    print(sess.run(v)) #输出0.099999905,即原来模型中变量v的滑动平均值

#下面代码给出了如何通过变量重命名直接读取变量的滑动平均值
#读取的变量v的值实际上是上面代码中变量v的滑动平均值
#通过这个方法就可以只用完全一样的代码来计算滑动平均模型前向传播的结果
v=tf.Variable(0,dtype=tf.float32,name="v")
#通过变量重命名将原来变量v的滑动平均值直接赋值给v
saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
with tf.Session() as sess:
    saver.restore(sess,"/path/to/model/model.ckpt")
    print(sess.run(v)) #输出0.099999905 这个值就是原来模型中变量v的滑动平均值



import tensorflow as tf
from tensorflow.python.framework import graph_util

v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0,shape=[2]),name="v2")

result=v1+v2

init_op=tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图的GraphDef部分，只需要这一个部分就可以完成从输入层到输出层的计算过程
    graph_def=tf.get_default_graph().as_graph_def()
    
    output_graph_def=graph_util.convert_variables_to_constants(sess,graph_def,['add'])
    #将导出的模型存入文件
    with tf.gfile.GFile("/path/to/model/combined_model.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())




#当只需要得到计算图中某个节点的取值时，这提供了一个更加方便的方法。
import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename="/path/to/model/combined_model.pb"
    #读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def,ParseFromString(f.read())
    #将graph_def中保存的图加载到当前的图中。
    #return_elements=["add":0]给出了返回的张量的名称。在保存
    #的时候给出的是计算节点的名称，所以为"add"。在加载的时候给出
    #的是张量的名称，所以是add:0
    result = tf.import_graph_def(graph_def,return_elements=["add:0"])
    print(sess.run(result))


import tensorflow as tf

#tf.train.NewCheckpointReader可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointReader('/path/to/model/model.ckpt')

#获取所有变量列表。这个事一个从变量名到变量维度的字典
all_variables=reader.get_variable_to_shape_map()
for variable_name in all_variables:
    #variable_name为变量名称，all_variable[variable_name]为变量的维度
    print(variable_name,all_variables[variable_name])
#获取名称为v1的变量
print("Value for variable v1 is",reader.get_tensor("v1"))




















