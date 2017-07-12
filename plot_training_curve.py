from matplotlib import pyplot
import numpy as np

def read_log(log_path):
    fid = open(log_path,'r')
    text = fid.readlines()
    if text.__len__()>2:
        train = text[0::2]
        val = text[1::2]
        trainErr = np.zeros([train.__len__(), 1])
        valErr = np.zeros([val.__len__(), 1])
        for lineT, lineV,ind in zip(train, val, range(train.__len__())):
            trainErr[ind] = float(lineT.split()[-1])
            valErr[ind] = float(lineV.split()[-1])
    else:
        train = text[0]
        val = text[1]
        trainErr = np.zeros([1, 1])
        valErr = np.zeros([1, 1])
        trainErr[0] = float(train.split()[-1])
        valErr[0] = float(val.split()[-1])

    return trainErr,valErr

def plot_training_curve(log_path, valIter):
    trainErr, valErr = read_log(log_path)
    iters = np.arange(trainErr.__len__())*valIter
    pyplot.figure(1)
    trainLine, = pyplot.plot(iters,trainErr,'r-o',linewidth=2.0,label='train')
    valLine, = pyplot.plot(iters,valErr,'b-x',linewidth=2.0,label='validation')
    if valErr.__len__()>1:
       valdiff = valErr[-2]-valErr[-1]
    else:
        valdiff = valErr
    pyplot.title('training curve')
    pyplot.legend(handles=[trainLine, valLine])
    pyplot.xlabel('iterations')
    pyplot.ylabel('loss')
    return pyplot.figure(1),valdiff,valErr[-1]