import csv
import time
import xlwt
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from unet import unet

def load_d_s(address, index, time, historical_data, skip_time, num_skip):
    dataset = np.load(address)
    i = 0
    for key in dataset.keys():
        i += 1
        if i == 1:
            d = dataset[key]
        else:
            d1 = dataset[key]
            d = np.concatenate([d, d1], axis=0)

    d = np.reshape(d, (-1,100,128,128))
    print("Data's shape: ", d.shape)
    x = d[index,time-skip_time*num_skip-historical_data+1:time-skip_time*num_skip+1]
    y = d[index,time]
    x = np.reshape(x,(-1,128,128,historical_data))
    y = np.reshape(y,(-1,128,128,1))

    return x, y

def load_d_c(address, time, historical_data, skip_time):
    dataset = np.load(address)
    i = 0
    for key in dataset.keys():
        i += 1
        if i == 1:
            d = dataset[key]
        else:
            d1 = dataset[key]
            d = np.concatenate([d, d1], axis=0)

    d = np.reshape(d, (-1,100,128,128))
    x = d[:,time-skip_time-historical_data+1:time-skip_time+1]
    y = d[:,time]
    x = np.reshape(x,(-1,128,128,historical_data))
    y = np.reshape(y,(-1,128,128,1))

    return x, y

def scale(data, max, min):
    d = data
    #d = (2*d-max-min)/(max-min)
    d = (d-min)/(max-min)

    return d

#show graph
def show_gra(x, address):
    x = np.reshape(x, (128,128,1))
    plt.imshow(x)
    plt.savefig(address, format='svg', dpi=800, transparent=True, bbox_inches='tight', pad_inches=0.0)

def cal_ME(pred, true):
    p = pred
    t = true
    num = tf.abs(p-t)**2
    num = tf.reduce_sum(num, axis=1)
    num = tf.reduce_sum(num, axis=1)
    den = tf.abs(t)**2
    den = tf.reduce_sum(den, axis=1)
    den = tf.reduce_sum(den, axis=1)
    r_err = num/den
    m_r_err = tf.reduce_mean(r_err, axis=0)

    return m_r_err

def np_64(x):

    return x.astype(np.float64)

def write_exl(data, savepath, filename):
    d = np.reshape(data, (128,128))
    d = np_64(d)

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet(filename, cell_overwrite_ok=True)
    col = ('x', 'y', 'value')
    
    for i in range(len(col)):
        sheet.write(0,i,col[i])
    
    row = 1
    for i in range(128):
        for j in range(128):
            sheet.write(row, 0, i)
            sheet.write(row, 1, j)
            sheet.write(row, 2, d[i][j])
            row += 1

    book.save(savepath)

def write_excel(savepath, time, error):
    d = np.zeros((len(time), 2))
    d[:,0] = time
    d[:,1] = error
    title = ["predicted time", "predicted error"]

    with open(savepath, "w", encoding="gbk", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(title)
        csv_writer.writerows(d)

def plot_graph(add_data,index_evolution,t,historical_data,skip_time,num_skip):
    #Loading the unet model
    model = unet()
    X_func = tf.random.normal([1,128,128,1],mean=0.0, stddev=1.0)
    model(X_func)
    model_address = "/user/best_model_unet.h5"
    model.load_weights(model_address)

    index = index_evolution
    h_d = historical_data
    s_t = skip_time
    n_s = num_skip
    add_d = add_data
    x, y = load_d_s(add_d, index, t, h_d, s_t, n_s)

    max_v = 1
    min_v = 0
    address = '/user/graph'
    y = scale(y, max_v, min_v)
    # show_gra(y, address+"/true.svg")
    write_exl(y, address+"/true.xls", "true")

    x = scale(x, max_v, min_v)
    # show_gra(x[-1], address+"/origin.svg")
    write_exl(x[-1], address+"/origin.xls", "origin")
    if h_d == 1:
        print("Input time: "+str(t-s_t*n_s))
    else:
        print("Input time: "+str(t-s_t*n_s-h_d+1)+" to "+str(t-s_t*n_s))

    begin_time = time.perf_counter()
    if h_d == 1:
        for _ in range(n_s):
            y_pred = model(x)
            x = np.reshape(y_pred, (-1,128,128,1))

        print("Elapsed time: "+f"{time.perf_counter()-begin_time:.6f}"+"s")
        # show_gra(y_pred, address+"/predicted.svg")
        write_exl(y_pred, address+"/predicted.xls", "predicted")

        point_error_test = abs(y_pred-y)
        # show_gra(point_error_test, address+"/point_error.svg")
        write_exl(point_error_test, address+"/point_error.xls", "point_error")
        print(model.loss_s(y_pred, y).numpy())
        print("Output time: "+str(t))

    # model.summary()
    print('--------Complete--------')

def cal_error(add_d,pred_start_t,pred_end_t,historical_data,skip_time):
    #Loading the unet model
    model = unet()
    X_func = tf.random.normal([1,128,128,1],mean=0.0, stddev=1.0)
    model(X_func)
    model_address = "/user/best_model_unet.h5"
    model.load_weights(model_address)
    model.summary()

    err = []
    time = list(range(pred_start_t, pred_end_t+1))
    for i in range(len(time)):
        x, y = load_d_c(add_d, time[i], historical_data, skip_time)

        max_v = 1
        min_v = 0
        x = scale(x, max_v, min_v)
        y = scale(y, max_v, min_v)

        batch_size = 2
        d_loss = 0.0
        n_d = 0
        for j in np.arange(batch_size, x.shape[0]+1, batch_size):
            start = j - batch_size
            x0 = x[start:j]
            y0 = y[start:j]

            y_pred = model(x0)
            d_loss += model.loss_s(y_pred, y0).numpy()
            # d_loss += cal_ME(y_pred, y0).numpy()
            n_d += 1
        
        d_loss /= n_d

        err.append(d_loss)

    prediction_add = "/user/prediction_unet.csv"
    write_excel(prediction_add, time, err)

if __name__ == "__main__":
    index = 12
    predicted_time = 90
    historical_data = 1
    skip_time = 5
    num_skip = 18        #The number of skips

    with tf.device("/GPU:1"):
        add_d = '/user/test.npz'
        plot_graph(add_d,index,predicted_time,historical_data,skip_time,num_skip)

        # add_d = '/user/test.npz'
        # pred_start_t = 5
        # pred_end_t = 99
        # cal_error(add_d,pred_start_t,pred_end_t,historical_data,skip_time)
