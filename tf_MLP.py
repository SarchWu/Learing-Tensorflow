import numpy as np
import tensorflow as tf
import os
import struct
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tf_CNN as cnn
tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
class Dataloader():
    def __init__(self,train_path,test_path):
        self.train_data_path=os.path.join(train_path,"train-images.idx3-ubyte")
        self.train_label_path=os.path.join(train_path,"train-labels.idx1-ubyte")
        self.test_data_path=os.path.join(test_path,"t10k-images.idx3-ubyte")
        self.test_label_path=os.path.join(test_path,"t10k-labels.idx1-ubyte")


    def load_mnist(self,filetype):
        if(filetype=='train'):
            with open(self.train_label_path, 'rb') as labelfilepath:
                labelfile=labelfilepath.read()
                head=struct.unpack_from('>II',labelfile,0)
                # print ("head:",head)
                imgnum=head[1]
                offset=struct.calcsize('>II')
                numString='>'+str(imgnum)+'B'
                labels=struct.unpack_from(numString,labelfile,offset)
                labels = np.reshape(labels,[imgnum])
                #print("training label shape",labels.shape,labels.dtype)
            with open(self.train_data_path,'rb') as imgfilepath:
                imagefile=imgfilepath.read()
                head=struct.unpack_from('>IIII',imagefile,0)
                # print ("head:",head)
                imgnum=head[1]
                img_cols=head[2]
                img_rows=head[3]
                offset=struct.calcsize('>IIII')
                numString='>'+str(imgnum*img_rows*img_cols)+'B'
                images=struct.unpack_from(numString,imagefile,offset)
                images=np.reshape(images,[imgnum,img_cols*img_rows])
                #print ("traing image shape: ",images.shape,images.dtype)
                # for i in images:
                #     print (i)
                #     i=np.reshape(i,[img_cols,img_rows])
                #     plt.imshow(i, 'gray')
                #     plt.pause(0.00001)
                #     plt.show()
            print ("image costs memory ：",images.nbytes)
            print ("label costs memory ：",labels.nbytes)
            return images,labels
        else:
            with open(self.test_label_path, 'rb') as labelfilepath:
                labelfile = labelfilepath.read()
                head = struct.unpack_from('>II', labelfile, 0)
                # print("head:", head)
                imgnum = head[1]
                offset = struct.calcsize('>II')
                numString = '>' + str(imgnum) + 'B'
                labels = struct.unpack_from(numString, labelfile, offset)
                labels = np.reshape(labels, [imgnum, ])
                #print ("test label shape: ",labels.shape)
                # print(labels)
            with open(self.test_data_path, 'rb') as imgfilepath:
                imagefile = imgfilepath.read()
                head = struct.unpack_from('>IIII', imagefile, 0)
                # print("head:", head)
                imgnum = head[1]
                img_cols = head[2]
                img_rows = head[3]
                offset = struct.calcsize('>IIII')
                numString = '>' + str(imgnum * img_rows * img_cols) + 'B'
                images = struct.unpack_from(numString, imagefile, offset)
                imgs = np.reshape(images, [imgnum, img_cols*img_rows])
                #print ("test image shape",imgs.shape)
            #     for i in imgs:
            #         print(i)
            #         i = np.reshape(i, [img_cols, img_rows])
            #         plt.imshow(i, 'gray')
            #         plt.pause(0.00001)
            #         plt.show()
            return imgs,labels
    def get_batch(self,batchsize,images,labels):
        # np.shape()[0] means getting the first dimension length
        index=np.random.randint(0,np.shape(images)[0],batchsize)
        #print (index)

        #print ("randomly choosed image's shape：",images[index,:].shape)
        #print("randomly choosed label's shape：", labels[index].shape)
        return images[index,:],labels[index]
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1=tf.keras.layers.Dense(units=100,activation=tf.nn.relu)
        self.dense2=tf.keras.layers.Dense(units=10)

    def __call__(self,inputs):
        x=self.dense1(inputs)
        x=self.dense2(x)
        return x

    def predict(self,inputs):
        logits=self(inputs)
        # tf.argmax返回最大值的索引号
        print (" logits shape: ",logits.shape)
        return tf.argmax(logits,axis=-1)




num_batches=1000
batch_size=5
learning_rate=0.001
#model=cnn.CNN()
model=MLP()
print ("init model down")
train_path=r"D:\deeplearning\data\train"
test_path=r"D:\deeplearning\data\val"
dataloader=Dataloader(train_path,test_path)
images,labels=dataloader.load_mnist('train')
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
for batch_index in range(num_batches):
    X,y=dataloader.get_batch(batch_size,images,labels)
    #print ("currently X,y shape and X,y dtype:",X.shape,y.shape,X.dtype,y.dtype,y)
    # test part
    #X_bak=np.reshape(X,[batch_size,28,28])
    #plt.imshow(X_bak, 'gray')
    #plt.pause(0.02)
    #plt.show()
    X=X.astype(np.float32)
    #y=y.astype(np.float32)
    #print ("currently X,y dtype and X,y shape:",X.dtype,y.dtype,X.shape,y.shape)
    with tf.GradientTape() as tape:
        y_logit_pred=model(tf.convert_to_tensor(X))
        print (y_logit_pred.shape,y_logit_pred.dtype)
        loss=tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_logit_pred)
        print ("batch：{}，loss：{}".format(batch_index,loss.numpy()))
    grads=tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

num_eval_samples=np.shape(dataloader.load_mnist('test')[0])[0]
print ("num_eval_samples: ",num_eval_samples)
y_pred=model.predict(dataloader.load_mnist('test')[0].astype(np.float32)).numpy()
print ("y_pred shape: ",y_pred.shape)
print("test accuracy:{}".format(sum(y_pred==(dataloader.load_mnist('test')[1]))/num_eval_samples))