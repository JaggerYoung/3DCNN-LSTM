import os,sys
import find_mxnet
import mxnet as mx
import numpy as np
import argparse
import logging
import cv2
import random
import glob

from symbol import c3d_bilstm

BATCH_SIZE = 2
LEN_SEQ = 10
SAMPLES = 30

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

	self.pad = 0
	self.index = None

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.label_names, self.label)]

def readData(Filename):
    data_1 = []
    data_2 = []
    f = open(Filename, 'r')
    total = f.readlines()
    print len(total)
    random.shuffle(total)

    for eachLine in range(len(total)):
        tmp = total[eachLine].split('\n')
	tmp_1, tmp_2 = tmp[0].split(' ',1)
	tmp_1 = '/data/zhigang.yang/UCF-101'+tmp_1
	data_1.append(tmp_1)
	data_2.append(int(tmp_2))

    f.close()
    return (data_1, data_2)

def ImageSeqToMatrix(dirName, length, samples, data_shape):
    pic = []
    for filename in glob.glob(dirName+'/*.jpg'):
        pic.append(filename)
    pic.sort()

    ret = []
    len_pic = len(pic)
   
    tmp = (len_pic-samples)/length
    for i in range(length):
        for j in range(samples):
	    index = i+j
	    ret.append(pic[i*tmp+j])

    r_1 = []
    g_1 = []
    b_1 = []
    mat = []

    for i in range(len(ret)):
        img = cv2.imread(ret[i], cv2.IMREAD_COLOR)
	b,g,r = cv2.split(img)
	r = cv2.resize(r, (data_shape[3], data_shape[2]))
	g = cv2.resize(g, (data_shape[3], data_shape[2]))
	b = cv2.resize(b, (data_shape[3], data_shape[2]))
	r = np.multiply(r, 1/255.0)
	g = np.multiply(g, 1/255.0)
	b = np.multiply(b, 1/255.0)
	r_1.append(r)
	g_1.append(g)
	b_1.append(b)

    mat.append(r_1)
    mat.append(g_1)
    mat.append(b_1)

    return mat

class UCFIter(mx.io.DataIter):
    def __init__(self, fname, num, batch_size, data_shape, init_states):
        self.batch_size = batch_size
	self.fname = fname
	self.data_shape = data_shape
	self.count = num/batch_size
	(self.data_1, self.data_2) = readData(fname)

	self.init_states = init_states
	self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

	self.provide_data = [('data', (batch_size,)+data_shape)]+init_states
	self.provide_label = [('label', (batch_size, ))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
        for k in range(self.count):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * batch_size + i
		pic = ImageSeqToMatrix(self.data_1[idx], LEN_SEQ, SAMPLES, self.data_shape)
		data.append(pic)
		#label_tmp = []
		#for j in range(LEN_SEQ):
		#    label_tmp.append(int(self.data_2[idx]))
	        #label.append(label_tmp)
	        label.append(int(self.data_2[idx]))

	    data_all = [mx.nd.array(data)]+self.init_state_arrays
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']+init_state_names
	    label_names = ['label']

	    data_batch =SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':

    train_num = 9537
    test_num = 3783

    num_hidden = 2048
    num_lstm_layer = 2

    num_label = 101
    seq_len = LEN_SEQ

    #train_num = 1820
    #test_num = 831

    batch_size = BATCH_SIZE
    data_shape = (3, LEN_SEQ*SAMPLES, 122, 122)
    num_label = 101

    devs = [mx.context.gpu(3)]
    network = c3d_bilstm(num_lstm_layer, seq_len, num_hidden, num_label)

    train_file = '/home/users/zhigang.yang/mxnet/example/3dcnn-bilstm-sm/data/train.list'
    test_file = '/home/users/zhigang.yang/mxnet/example/3dcnn-bilstm-sm/data/test.list'

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

    data_train = UCFIter(train_file, train_num, batch_size, data_shape, init_states)
    data_val = UCFIter(test_file, test_num, batch_size, data_shape, init_states)
    print data_train.provide_data,data_train.provide_label

    model = mx.model.FeedForward(ctx           = devs,
                                 symbol        = network,
                                 num_epoch     = 500,
                                 learning_rate = 0.003,
                                 momentum      = 0.9,
                                 wd            = 0.005,
                                 initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    batch_end_callbacks = [mx.callback.Speedometer(BATCH_SIZE, 1000)]
    print 'begin fit'

    eval_metrics = ['accuracy']
    model.fit(X = data_train, eval_data = data_val, eval_metric = eval_metrics, batch_end_callback = batch_end_callbacks)
