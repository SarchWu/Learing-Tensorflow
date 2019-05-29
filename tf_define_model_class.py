import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()
X=tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 初始化类的全连接层
        self.dense=tf.keras.layers.Dense(units=1,kernel_initializer=tf.zeros_initializer,bias_initializer=tf.zeros_initializer)
    #  定义一个类的函数
    def __call__(self,input):
        output=self.dense(input)
        return output

model=Linear()
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-3)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred=model(X)
        loss=tf.reduce_mean(tf.square(y_pred-y))
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))
print (model.variables)
