import tensorflow as tf
from array import array
import mnist as m
from PIL import Image
import numpy as np

mnist = m.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for i in range(1000):
    print (i/10)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, {x: batch_xs, y_: batch_ys})



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels}))

list = array('f')
for i in range(0,784) :
    if (i%28 == 14):
        list.append(1.)
    else:
        list.append(0.)


img = Image.open('MYDATA/6_1.png').convert('RGBA')
a = np.array(img)
fun = lambda t: 1-(t/255)
vfunc = np.vectorize(fun)
six = vfunc(a)

test = [six.tolist()]

arg = tf.argmax(y,1)
print(sess.run([arg,y],{x:test}))
