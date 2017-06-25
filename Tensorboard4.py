'''
Created on 2017年5月23日

@author: weizhen
'''
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# minist_inference中定义的常量和前向传播的函数不需要改变，
# 因为前向传播已经通过tf.variable_scope实现了计算节点按照网络结构的划分
import mnist_inference
from mnist_train import MOVING_AVERAGE_DECAY, REGULARAZTION_RATE, \
    LEARNING_RATE_BASE, BATCH_SIZE, LEARNING_RATE_DECAY, TRAINING_STEPS, MODEL_SAVE_PATH, MODEL_NAME
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
def train(mnist):
    # 将处理输入数据集的计算都放在名子为"input"的命名空间下
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-cinput')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)
    
    # 将滑动平均相关的计算都放在名为moving_average的命名空间下
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
    
    # 将计算损失函数相关的计算都放在名为loss_function的命名空间下
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        
    # 将定义学习率、优化方法以及每一轮训练需要执行的操作都放在名子为"train_step"的命名空间下
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                 global_step,
                                                 mnist.train._num_examples / BATCH_SIZE,
                                                 LEARNING_RATE_DECAY,
                                                 staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    
    # 训练模型。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            if i % 1000 == 0:
                # 配置运行时需要记录的信息。
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto。
                run_metadata = tf.RunMetadata()
                _, loss_value, step = sess.run(
                    [train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                    options=run_options, run_metadata=run_metadata)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                writer = tf.summary.FileWriter("/log/modified_mnist_train.log", tf.get_default_graph())
                writer.add_run_metadata(run_metadata, "stop%03d" % i)
                writer.close()
                print("After %d training steps(s),loss on training batch is %g."%(step,loss_value))
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
    # 初始化Tensorflow持久化类
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #    tf.global_variables_initializer().run()
    #    
        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序来完成
    #    for i in range(TRAINING_STEPS):
    #        xs, ys = mnist.train.next_batch(BATCH_SIZE)
    #        _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
            
            # 每1000轮保存一次模型
    #        if i % 1000 == 0:
                # 输出当前训练情况。这里只输出了模型在当前训练batch上的损失函数大小
                # 通过损失函数的大小可以大概了解训练的情况。在验证数据集上的正确率信息
                # 会有一个单独的程序来生成
    #            print("After %d training step(s),loss on training batch is %g" % (step, loss_value))
                
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件末尾加上训练的轮数
                # 比如"model.ckpt-1000"表示训练1000轮之后得到的模型
    #            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    
                # 将当前的计算图输出到TensorBoard日志文件
    #            writer=tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
    #            writer.close()
    

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
