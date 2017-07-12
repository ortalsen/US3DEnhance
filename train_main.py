import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import h5py
import os
from v_net import V_net
import plot_training_curve

device = '/gpu:0'
stoppingCriteria = 1e-5
earlyStoppingIters = 10
##################################################### DATA #############################################################
h = 128
w = 128
depth = 64
num_channels = 2
num_out_channels = 1
dataPath = '/home/ortalsenouf/Documents/Data/3DUS'
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

##################################################Constructing net architecture#########################################
with tf.device(device):
    x = tf.placeholder(tf.float32, shape=[2, depth, h, w, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[2, depth, h, w, num_out_channels], name='y_true')
    y_estim = V_net(x,kernel_size=5,initializer=tf.contrib.layers.variance_scaling_initializer(),nonlinearity=tf.nn.relu)
    MSE = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(y_estim - y_true), 1), 1), 1)
    cost = tf.reduce_mean(MSE)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
session.run(tf.initialize_all_variables())
##################################### optimization helper function #####################################################
def shuffle_data(indTrain):
        ind = np.random.permutation(np.range(indTrain.__len__()))
        indTrain = indTrain(ind)
        return indTrain
##################################### optimization helper function ###############################################
def print_valid_MSE(test_batch_size,num_test,indValid,log_file):
    i = 0
    MSEtot = np.zeros([num_test,1])
    x_batch = np.zeros([test_batch_size, depth, h, w, num_channels])
    y_true_batch = np.zeros([test_batch_size, depth, h, w, num_out_channels])
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        batch_files = DSfiles[indValid[i:j]]
        for k in range(batch_files.__len__()):
            fid = h5py.File(dataPath + '/' + batch_files[k], 'r')
            x_batch[k, :, :, :, :] = np.transpose(np.array(fid['data']), [4, 3, 2, 1, 0])
            y_true_batch[k, :, :, :, :] = np.expand_dims(np.transpose(np.array(fid['label']), [2, 1, 0, 3]), 0)
        feed_dict = {x: x_batch,
                     y_true: y_true_batch}
        MSEbatch = session.run(MSE, feed_dict=feed_dict)
        MSEtot[i:j] = MSEbatch
        i = j
    mean_MSE = np.mean(MSEtot)
    pSNR = 20*np.log10(1/np.sqrt(MSEtot))
    mean_pSNR = np.mean(pSNR)

    msg = "mean MSE on Validation-Set: {0}"
    print(msg.format(mean_MSE))
    log_file.write(msg.format(mean_MSE)+'\n')

    msg = "mean pSNR on Validation-Set: {0}"
    print(msg.format(mean_pSNR))
    log_file.write(msg.format(mean_pSNR)+'\n')
def optimize(num_iterations,train_batch_size,indTrain,indValid,dataPath):
    start_time = time.time()
    j = 0
    x_batch = np.zeros([train_batch_size,depth,h,w,num_channels])
    y_true_batch = np.zeros([train_batch_size,depth,h,w,num_out_channels])
    valMSE=1e7
    criteriaCheck=0
    for i in range(num_iterations):
        if j*train_batch_size+train_batch_size <= num4Train:
            batch_files = DSfiles[indTrain[(j*train_batch_size):(j*train_batch_size+train_batch_size)]]
            for k in range(batch_files.__len__()):
                fid = h5py.File(dataPath+'/'+batch_files[k],'r')
                x_batch[k,:,:,:,:] = np.transpose(np.array(fid['data']),[4,3,2,1,0])
                y_true_batch[k,:,:,:,:] = np.expand_dims(np.transpose(np.array(fid['label']),[2,1,0,3]), 0)
            j+=1
        else:
            j = 0
            batch_files = DSfiles[indTrain[(j * train_batch_size):(j * train_batch_size + train_batch_size)]]
            for k in range(batch_files.__len__()):
                fid = h5py.File(dataPath + '/' + batch_files[k], 'r')
                x_batch[k, :, :, :, :] = np.transpose(np.array(fid['data']), [4, 3, 2, 1, 0])
                y_true_batch[k, :, :, :, :] = np.expand_dims(np.transpose(np.array(fid['label']), [2, 1, 0, 3]), 0)
            j+=1
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(train,feed_dict=feed_dict_train)
        if i % 1000 == 0:
            log_file = open('log.txt', 'a')
            MSE = session.run(cost, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0}, Training loss: {1}"
            print(msg.format(i + 1, MSE))
            log_file.write(msg.format(i + 1, MSE)+'\n')
            print_valid_MSE(test_batch_size=2, num_test=num4Val, indValid=indValid,log_file =log_file)
            log_file.close()
            if i > 999:
                fig.clf()
            fig, valdiff, valErr = plot_training_curve.plot_training_curve(
                '/home/ortalsenouf/Documents/logs/Mars22_17/log_sep2.txt', 640)
            valdiff = 1
            if valErr<valMSE:
                valMSE=valErr
                saver.save(session, '/home/ortalsenouf/Documents/checkpoints/June4/iter' + str(i))
            if valdiff <= stoppingCriteria or valdiff < 0:
                criteriaCheck += 1
            fig.show()
            fig.savefig('/home/ortalsenouf/Documents/logs/Mars22_17/training_curve_sep2.png')
            if criteriaCheck >= earlyStoppingIters:
                break
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    log_file = open('log.txt','a')
    log_file.write("Time usage: " + str(timedelta(seconds=int(round(time_dif)))) + '\n')
    log_file.close()
#######################Run Optimization and Test##############################
saver = tf.train.Saver()

optimize(num_iterations=np.int(1e6), train_batch_size=2,indTrain=indTrain,indValid=indVal, dataPath=dataPath)

print('ok')
