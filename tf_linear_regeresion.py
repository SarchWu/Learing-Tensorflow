from PIL import Image
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()


'''
constant:常量
variable：变量，变量需要给它赋初值
example：one_layer
'''
A=tf.constant(1)
B=tf.constant(1)
C=tf.add(A,B)
# print (C)
# 记住array都要用最外面一个[]包括进来
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[5,6],[7,8]])
c=tf.matmul(a,b)
# print(c)
# 创建一个变量并且进行初始化
x=tf.get_variable('x',shape=[1],initializer=tf.constant_initializer(4.))
# with 内构建需要求导的部分
with tf.GradientTape() as tape:
    y=tf.square(x)
y_grad=tape.gradient(y,x)
# print ([y.numpy(),y_grad.numpy()])


# 已知X,y,初始化w/b,求参数w,b,以及w_grad，b_grad，不涉及参数更新
'''
对谁求偏导时，谁就是变量
其余不变的为常量
'''
X=tf.constant([[1.,2.],[3.,4.]])
y=tf.constant([[1.],[2.]])
w=tf.get_variable('w',shape=[2,1],initializer=tf.constant_initializer([[1.],[2.]]))
b=tf.get_variable('b',shape=[1],initializer=tf.constant_initializer([1.]))
with tf.GradientTape() as tape:
    L=0.5*tf.reduce_sum(tf.square(tf.matmul(X,w)+b-y))
w_grad,b_grad=tape.gradient(L,[w,b])
# print([L.numpy(),w_grad.numpy(),b_grad.numpy()])

'''
using numpy do linear regression
'''

# 已知X，y,初始化a，b，手动设置求偏导公式以及梯度下降（参数更新）公式
import numpy as np
X_raw=np.array([2013,2014,2015,2016,2017])
y_raw=np.array([12000,14000,15000,16500,17500])
# 数据归一化
X=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
y=(y_raw-y_raw.min())/(y_raw.max()-y_raw.min())
# print (X,y)
# 参数初始化
a = np.array([0])
b= np.array([0])
# loss函数求偏导，根据偏导结果求极小值时的a,b
num_epoch=30000
lr=1e-3
for r in range(num_epoch):
    # 训练的过程，就是初始化a,b之后，输入自变量，得到预测因变量，判断预测因变量和实际因变量的差距的过程
    y_pred=a*X+b
    #print ("type y_pred : ",y_pred)
    # 这里为什么要加dot和sum,把输入和输出当成向量看就好了
    # 一次性投入多个数据求解梯度
    grad_a,grad_b=(y_pred-y).dot(X),(y_pred-y).sum()
    # print (grad_a,grad_b)
    # 为什么是-，根据偏导时切线方向以及极值点方向确定的
    # 更新参数
    a,b=a-lr*grad_a,b-lr*grad_b
    # print (a,b)


'''
# tape.gradient(ys,xs)自动计算梯度
# optimizer.apply_gradients(grads_and_vars)自动更新模型参数
# 已知X，y，初始化a，b（zero_optimizer）,自动计算梯度，自动更新参数
'''
X=tf.constant(X,dtype=tf.float32)
y=tf.constant(y,dtype=tf.float32)
#print (X,y)
a=tf.get_variable('a',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer)
b=tf.get_variable('b',dtype=tf.float32,shape=[],initializer=tf.zeros_initializer)
#print (a,b)
variables=[a,b]
num_epoch=10000
# 创建一个优化对象
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred=a*X+b
        loss=0.5*tf.reduce_sum(tf.square(y_pred-y))
        print (loss)
    # 计算loss关于自变量的梯度（偏导），输入loss函数以及自变量
    grads=tape.gradient(loss,variables)
    # 此时参数（也就是自变量）尚未更新
    #print (grads,variables)
    # 更新参数,grads_and_vars，待更新的自变量偏导数以及自变量对，zip一对一打包成元组
    optimizer.apply_gradients(grads_and_vars=zip(grads,variables))
    #print (grads,variables)
