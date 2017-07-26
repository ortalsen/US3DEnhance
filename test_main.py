import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import tensorflow as tf
import numpy as np
import h5py
import os
from v_net import V_net
from scipy.io import savemat


##################################################### DATA #############################################################
h = 128
w = 128
depth = 64
num_channels = 2
num_out_channels = 1
dataPath = '/home/ortals/data/3DUS'
DSfiles = np.array(os.listdir(dataPath))
numOfSamps = DSfiles.__len__()
trainRatio = 0.8
TestRatio = 0.2
ValRatio = 0.2
trainRatio = trainRatio*(1-ValRatio)
ValRatio = 1-trainRatio-TestRatio
num4Train = int(np.round(trainRatio*numOfSamps))
num4Val = int(np.round(ValRatio*numOfSamps))
num4Test = int(np.round(TestRatio*numOfSamps))
indTot = np.random.permutation(numOfSamps)
indTrain = indTot[0:num4Train]
indVal = indTot[num4Train:(num4Train+num4Val)]
indTest = indTot[(num4Train+num4Val):]
np.save('indTest',indTest)
##################################################Constructing net architecture#########################################
x = tf.placeholder(tf.float32, shape=[2, depth, h, w, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[2, depth, h, w, num_out_channels], name='y_true')
y_estim = V_net(x,kernel_size=5,initializer=tf.contrib.layers.variance_scaling_initializer(),nonlinearity=tf.nn.relu)
MSE = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(y_estim - y_true), 1), 1), 1)
cost = tf.reduce_mean(MSE)

session = tf.Session()

saver = tf.train.Saver()

i = 0
ckp = '/home/ortals/Results/checkpoints/3DUS/19071737119'
saver.restore(session, ckp)
test_batch_size = 2
num4Test = 20
MSEtot = np.zeros([num4Test, 1])
x_batch = np.zeros([test_batch_size, depth, h, w, num_channels])
y_true_batch = np.zeros([test_batch_size, depth, h, w, num_out_channels])
indTest = np.load('indTest.npy')
while i < num4Test:
    j = min(i + test_batch_size, num4Test)
    batch_files = DSfiles[indTest[i:j]]
    for k in range(batch_files.__len__()):
        fid = h5py.File(dataPath + '/' + batch_files[k], 'r')
        x_batch[k, :, :, :, :] = np.transpose(np.array(fid['data']), [4, 3, 2, 1, 0])
        y_true_batch[k, :, :, :, :] = np.expand_dims(np.transpose(np.array(fid['label']), [2, 1, 0, 3]), 0)
    feed_dict = {x: x_batch,
                     y_true: y_true_batch}
    MSEbatch,Im3D = session.run([MSE,y_estim], feed_dict=feed_dict)
    MSEtot[i:j] = MSEbatch
    mat_dict = {'labels': y_true_batch, 'estim':Im3D}
    savemat('/home/ortals/Results/images/V-net2/'+str(2*i+1)+str(2*i+2)+'.mat',mat_dict)
    i = j
mean_MSE = np.mean(MSEtot)

print('ok')
