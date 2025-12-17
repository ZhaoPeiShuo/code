import numpy as np
import tensorflow as tf
import time
import sys
import random
import matplotlib.pyplot as plt

from unet import unet
from lion_tf1 import Lion

import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

@tf.function
def train(model, x, y, optimizer):
    
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = model.loss(y_pred, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return(loss)

@tf.function
def distributed_train_step(model, x, y, optimizer):
    loss = strategy.run(train, args=(model, x, y, optimizer))

    return strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)

@tf.function
def distributed_test_step(y_pred, y_true, loss_type):
    loss = strategy.run(loss_type, args=(y_pred, y_true))

    return strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)

def split_data(data, historical_data, skip_time):
    length = data.shape[1]
    h_d = historical_data
    s_t = skip_time
    x = data[:,:length-s_t]
    y = data[:,h_d+s_t-1:]
    x = np.transpose(x, axes=[0, 2, 3, 1])
    y = np.transpose(y, axes=[0, 2, 3, 1])
    
    dim1 = x.shape[0]
    dim2 = x.shape[1]
    dim3 = x.shape[2]
    dim4 = x.shape[3]
    x = np.reshape(x, (dim1,dim2,dim3,dim4,1))

    return x, y

def rotate(arr):
    arr = np.array(arr)
    if len(arr.shape) != 4:
        string = "Wrong! The array is not two-dimensional."
        return (string)
    else:
        length1 = arr.shape[0]
        length2 = arr.shape[1]
        dim_1 = arr.shape[2]
        dim_2 = arr.shape[3]
        r_arr = np.zeros(shape=(length1, length2, dim_2, dim_1),dtype=np.float32)
        for i in range(dim_1):
            a = arr[:,:,i,:]
            r_arr[:,:,:,-(i+1)] += a
        
        return r_arr

def flip1(arr):         # Flip images side to side
    length1 = arr.shape[0]
    length2 = arr.shape[1]
    dim_1 = arr.shape[2]
    dim_2 = arr.shape[3]
    f_arr = np.zeros(shape=(length1, length2, dim_1, dim_2),dtype=np.float32)
    for i in range(dim_1):
        a = arr[:,:,i,:]
        f_arr[:,:,-(i+1),:] += a
    
    return f_arr

def flip2(arr):         # Flip images back and forth
    length1 = arr.shape[0]
    length2 = arr.shape[1]
    dim_1 = arr.shape[2]
    dim_2 = arr.shape[3]
    f_arr = np.zeros(shape=(length1, length2, dim_1, dim_2),dtype=np.float32)
    for i in range(dim_2):
        a = arr[:,:,:,i]
        f_arr[:,:,:,-(i+1)] += a
    
    return f_arr
    
def expansion(data):
    d = data
    a = 63

    q1 = d[:, :, -a:, -a:]
    q2 = d[:, :, -a:, :]
    q3 = d[:, :, -a:, :a]
    q4 = d[:, :, :, -a:]
    q5 = d
    q6 = d[:, :, :, :a]
    q7 = d[:, :, :a, -a:]
    q8 = d[:, :, :a, :]
    q9 = d[:, :, :a, :a]

    expanded1 = tf.concat([q1, q4, q7], axis=2)
    expanded2 = tf.concat([q2, q5, q8], axis=2)
    expanded3 = tf.concat([q3, q6, q9], axis=2)
    expanded = tf.concat([expanded1, expanded2, expanded3], axis=3)

    return expanded

def augment_data(data, x, y):
    d = data
    length = d.shape[0]
    dim1 = len(x)*len(y)
    a_d = np.zeros(shape=(dim1,length,d.shape[1],128,128), dtype=np.float32)

    index = 0
    for i in range(len(x)):
        for j in range(len(y)):
            a_d[index]=d[:,:,x[i]:x[i]+128,y[j]:y[j]+128]
            index += 1

    a_d = np.reshape(a_d,(-1,d.shape[1],128,128))

    return a_d

def load_d(add_d):
    dataset = np.load(add_d)
    i = 0
    for key in dataset.keys():
        i += 1
        if i == 1:
            d = dataset[key]
        else:
            d1 = dataset[key]
            d = np.concatenate([d, d1], axis=0)

    min = 0
    max = 1
    d = (d-min)/(max-min)

    return d

def train_d(model, x_train, y_train, batch_size, historical_data, optimizer):
    train_loss = 0.0
    n_train = 0
    h_d = historical_data
    for end in np.arange(batch_size, x_train.shape[0]+1, batch_size):
        start = end - batch_size
        for j in range(0, y_train.shape[-1]):
            x = x_train[start:end,:,:,j:j+h_d]
            dim = x.shape[-1]
            x = np.reshape(x, (-1,128,128,h_d*dim))
            y = y_train[start:end,:,:,j]
            train_loss += distributed_train_step(model, x, y, optimizer)
            n_train += 1

    train_loss /= n_train

    return train_loss

def test_d(model, x_test, y_test, batch_size, historical_data):
    test_loss = 0.0
    n_test = 0
    h_d = historical_data
    for end in np.arange(batch_size, x_test.shape[0]+1, batch_size):
        start = end - batch_size
        for j in range(0, y_test.shape[-1]):
            x = x_test[start:end,:,:,j:j+h_d]
            dim = x.shape[-1]
            x = np.reshape(x, (-1,128,128,h_d*dim))
            y_pred = model(x)
            test_loss += distributed_test_step(y_pred, y_test[start:end,:,:,j], model.loss_s)
            n_test += 1

    test_loss /= n_test

    return test_loss

def cal_first_derivative_x(data):
    d = data
    d1 = np.zeros_like(d)
    d2 = np.zeros_like(d)
    
    d1[:,:,:127] = d[:,:,1:]
    d1[:,:,127] = d[:,:,0]
    d2[:,:,0] = d[:,:,127]
    d2[:,:,1:] = d[:,:,:127]

    d_x = (d1-d2)/2

    return d_x

def cal_first_derivative_y(data):
    d = data
    d1 = np.zeros_like(d)
    d2 = np.zeros_like(d)

    d1[:,:127] = d[:,1:]
    d1[:,127] = d[:,0]
    d2[:,0] = d[:,127]
    d2[:,1:] = d[:,:127]

    d_y = (d1-d2)/2

    return d_y

def add_gaussian(x, y, times=1, mean=0.0, stddev=0.01):
    if times == 1:
        return x, y
    elif times > 1:
        x_ans = x
        y_ans = y
        for _ in range(times-1):
            gau_noise = np.random.normal(loc=mean, scale=stddev, size=x.shape)
            x_ans = np.concatenate([x_ans, x+gau_noise], axis=0)
            y_ans = np.concatenate([y_ans, y], axis=0)
        return x_ans, y_ans
    else:
        print("The value of parameter 'times' can only be an integer and larger than 0.")
        return 0, 0

def train_mode(model, x_train, y_train, x_test, y_test, batch_size_train, batch_size_test, historical_data, optimizer, 
               model_add, data_add, fine_tuning=0, model_f_add="/None", data_f_add="/None", early_stop=1, num_epoch=1, Num_stop=1):
    begin_time = time.time()

    if fine_tuning == 1:
        X_func = tf.random.normal([1,128,128,1],mean=0.0, stddev=1.0)
        model(X_func)
        model.load_weights(model_f_add)

        index_list = np.load(data_f_add)["index_list"]
        train_loss_list = np.load(data_f_add)["train_loss_list"]
        test_loss_list = np.load(data_f_add)["test_loss_list"]
        index_list = list(index_list)
        train_loss_list = list(train_loss_list)
        test_loss_list = list(test_loss_list)
    elif fine_tuning == 0:
        index_list = []
        train_loss_list = []
        test_loss_list = []
    else:
        print("Wrong! The value of parameter 'fine_tuning' can only be 0 or 1.")

    if early_stop == 1:
        len_original_epochs = len(index_list)
        num_stop = 0
        i = 0
        while num_stop<Num_stop:
            train_loss = train_d(model, x_train, y_train, batch_size_train, historical_data, optimizer)
            test_loss = test_d(model, x_test, y_test, batch_size_test, historical_data)

            if i < 1:
                mid_loss = test_loss
                best_epoch = 0
                model.save_weights(model_add)
            else:
                    
                if mid_loss > test_loss:
                    num_stop = 0
                    best_epoch = i
                    mid_loss = test_loss
                    model.save_weights(model_add)
                else:
                    num_stop += 1
            
            print("epoch:" + str(i)+", Train Loss:"+"{:.3e}".format(train_loss)+", Test Loss:"+"{:.3e}".format(test_loss)+", elapsed time: "+str(int(time.time()-begin_time))+"s"  )

            index_list.append(i+len_original_epochs)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            i += 1

        print("Best epoch: "+str(best_epoch+len_original_epochs)+", Train loss:"+"{:.3e}".format(train_loss_list[best_epoch+len_original_epochs])+", Test loss:"+"{:.3e}".format(test_loss_list[best_epoch+len_original_epochs]))
        np.savez(data_add, index_list=index_list, train_loss_list=train_loss_list, test_loss_list=test_loss_list)
    elif early_stop == 0:
        len_original_epochs = len(index_list)
        for i in range(num_epoch):
            train_loss = train_d(model, x_train, y_train, batch_size_train, historical_data, optimizer)
            test_loss = test_d(model, x_test, y_test, batch_size_test, historical_data)

            if i < 1:
                mid_loss = test_loss
                best_epoch = 0
                model.save_weights(model_add)
            else:
                if mid_loss > test_loss:
                    best_epoch = i
                    mid_loss = test_loss
                    model.save_weights(model_add)
            
            print("epoch:" + str(i)+", Train Loss:"+"{:.3e}".format(train_loss)+", Test Loss:"+"{:.3e}".format(test_loss)+", elapsed time: "+str(int(time.time()-begin_time))+"s"  )

            index_list.append(i+len_original_epochs)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)

        print("Best epoch: "+str(best_epoch+len_original_epochs)+", Train loss:"+"{:.3e}".format(train_loss_list[best_epoch+len_original_epochs])+", Test loss:"+"{:.3e}".format(test_loss_list[best_epoch+len_original_epochs]))
        np.savez(data_add, index_list=index_list, train_loss_list=train_loss_list, test_loss_list=test_loss_list)
    else:
        print("Wrong! The value of parameter 'early_stop' can only be 0 or 1.")

    return index_list, train_loss_list, test_loss_list

def plot_graph(index_list, train_loss_list, test_loss_list, graph_add):
    fig = plt.figure(figsize=(10,7))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, test_loss_list, label="test", linewidth=2)
    plt.legend(fontsize=16)
    plt.yscale('log')
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("MSE", fontsize=18)
    plt.savefig( graph_add, dpi=800)

    return 1

if __name__ == "__main__":

    #Hyperparameters
    seeds = 13686
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seeds)
    random.seed(seeds)
    np.random.seed(seeds)
    tf.random.set_seed(seeds)
    
    historical_data = 1
    skip_time = 5
    batch_size_train = 32
    batch_size_test = 16
    l_r = 0.0001       #learning rate 0.00001
    Num_stop = 10
    cut_time = 0
    fine_tuning = 0     # Use trained parameters as initialization parameters or not
    early_stop = 1      # Use early stop strategy or not
    num_epoch = 200     # The number of training epochs

    #Load dataset
    path = '/user'
    path_x_train = path+"/data/train.npz"
    path_x_test = path+"/data/test.npz"
    d_train = load_d(path_x_train)
    d_test = load_d(path_x_test)

    #cut time
    d_train = d_train[:,cut_time:]

    ## Augment data
    # Method 1
    # x_points = np.random.choice(np.arange(0,63*2), 2, replace=False)
    # y_points = np.random.choice(np.arange(0,63*2), 2, replace=False)
    # d_train = expansion(d_train)
    # d_train = augment_data(d_train, x_points, y_points)

    # Method 2
    # x_times = 4
    # y_times = 4
    # x_points = np.array(range(0,128,int(np.floor(128/x_times))))
    # y_points = np.array(range(0,128,int(np.floor(128/y_times))))
    # x_points = x_points[:x_times]
    # y_points = y_points[:y_times]
    # d_train = expansion(d_train)
    # d_train = augment_data(d_train, x_points, y_points)

    # Through rotating graphs to augment dataset
    # d_r1 = rotate(d_train)
    # d_r2 = rotate(d_r1)
    # d_r3 = rotate(d_r2)
    # d_train = np.concatenate([d_train, d_r1, d_r2, d_r3], axis=0)
    # del d_r1, d_r2, d_r3

    # Through flipping graphs to augment dataset
    # d_r1 = flip1(d_train)
    # d_r2 = flip2(d_r1)
    # d_r3 = flip1(d_r2)
    # d_train = np.concatenate([d_train, d_r1, d_r2, d_r3], axis=0)
    # del d_r1, d_r2, d_r3

    x_train, y_train = split_data(d_train, historical_data, skip_time)
    x_test, y_test = split_data(d_test, historical_data, skip_time)
    del d_train, d_test

    #Add gradient information
    # x_train_x = cal_first_derivative_x(x_train)
    # x_train_y = cal_first_derivative_y(x_train)
    # x_train = tf.concat([x_train, x_train, x_train], axis=4)
    # del x_train_x, x_train_y

    # x_test_x = cal_first_derivative_x(x_test)
    # x_test_y = cal_first_derivative_y(x_test)
    # x_test = tf.concat([x_test, x_test_x, x_test_y], axis=4)
    # del x_test_x, x_test_y

    # Through adding gaussian noise to augment dataset
    # x_train, y_train = add_gaussian(x_train, y_train, times=16, mean=0.0, stddev=0.1)

    # shuffler = np.random.permutation(x_train.shape[0])
    # x_train = x_train[shuffler]
    # y_train = y_train[shuffler]
    # del shuffler
    
    print("The shape of x_train: ", x_train.shape)
    print("The shape of y_train: ", y_train.shape)
    print("The shape of x_test: ", x_test.shape)
    print("The shape of y_test: ", y_test.shape)

    #Create an object
    with strategy.scope():
        model = unet()
        print('Model created')
        # optimizer = Lion(learning_rate = l_r)
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_r)

        model.compile(optimizer=optimizer, loss=model.loss)

    model_add = path+"/data/best_model_unet"+".h5"
    data_add = path+'/data/convergence_data_unet.npz'
    print('Training Begins')

    index_list, train_loss_list, test_loss_list = train_mode(model, x_train, y_train, x_test, y_test, 
                                                            batch_size_train, batch_size_test, historical_data, 
                                                            optimizer, model_add, data_add, fine_tuning=fine_tuning, 
                                                            early_stop=early_stop, num_epoch=num_epoch, Num_stop=Num_stop)

    graph_add = path+"/graph"
    plot_graph(index_list, train_loss_list, test_loss_list, graph_add)

    print('--------Complete--------')