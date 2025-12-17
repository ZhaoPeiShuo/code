import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.layers import Reshape, Conv2D, AveragePooling2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, Conv2DTranspose, Add, BatchNormalization, Concatenate, Input
from tensorflow.keras.utils import get_custom_objects

#Defining activation function
def act_func1(x):
    y = tf.nn.relu(x)
    return y

def act_func2(x):
    y = tf.tanh(x)
    return y

def act_func3(x):
    y = tf.sigmoid(x)
    return y

get_custom_objects().update({'act_func1': Activation(act_func1)})
get_custom_objects().update({'act_func2': Activation(act_func2)})
get_custom_objects().update({'act_func3': Activation(act_func3)})

class unet(tf.keras.Model):

    def __init__(self):
        
        self.scale = 1
        self.lamda = 0.001
        self.die_rate = 0.0 #0.4
        self.initializer1 = tf.keras.initializers.HeNormal()
        self.initializer2 = tf.keras.initializers.GlorotUniform()
        
        super(unet, self).__init__()
        self.seeds = 13686
        np.random.seed(self.seeds)
        tf.random.set_seed(self.seeds)
        
        self.unet = self.build_unet()
    
    def expansion(self, x, size):
        a = (size - 1) // 2

        q1 = x[:, -a:, -a:]
        q2 = x[:, -a:, :]
        q3 = x[:, -a:, :a]
        q4 = x[:, :, -a:]
        q5 = x
        q6 = x[:, :, :a]
        q7 = x[:, :a, -a:]
        q8 = x[:, :a, :]
        q9 = x[:, :a, :a]

        expanded1 = tf.concat([q1, q4, q7], axis=1)
        expanded2 = tf.concat([q2, q5, q8], axis=1)
        expanded3 = tf.concat([q3, q6, q9], axis=1)
        expanded = tf.concat([expanded1, expanded2, expanded3], axis=2)

        return expanded
        


    def build_unet(self):
        input = Input(shape=((128,128,1)))

        e1_1 = self.expansion(input, 3)
        conv1_1 = Conv2D(16*self.scale, 3, padding='valid', name='conv1_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e1_1)
        bn1_1 = BatchNormalization(name='bn1_1')(conv1_1)
        r1_1 = Activation('act_func1')(bn1_1)
        e1_2 = self.expansion(r1_1, 3)
        conv1_2 = Conv2D(16*self.scale, 3, padding='valid', name='conv1_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e1_2)
        bn1_2 = BatchNormalization(name='bn1_2')(conv1_2)
        r1_2 = Activation('act_func1')(bn1_2)
        p1 = MaxPooling2D(pool_size = 2, strides = 2)(r1_2)

        e2_1 = self.expansion(p1, 3)
        conv2_1 = Conv2D(32*self.scale, 3, padding='valid', name='conv2_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e2_1)
        bn2_1 = BatchNormalization(name='bn2_1')(conv2_1)
        r2_1 = Activation('act_func1')(bn2_1)
        e2_2 = self.expansion(r2_1, 3)
        conv2_2 = Conv2D(32*self.scale, 3, padding='valid', name='conv2_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e2_2)
        bn2_2 = BatchNormalization(name='bn2_2')(conv2_2)
        r2_2 = Activation('act_func1')(bn2_2)
        p2 = MaxPooling2D(pool_size = 2, strides = 2)(r2_2)

        e3_1 = self.expansion(p2, 3)
        conv3_1 = Conv2D(64*self.scale, 3, padding='valid', name='conv3_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e3_1)
        bn3_1 = BatchNormalization(name='bn3_1')(conv3_1)
        r3_1 = Activation('act_func1')(bn3_1)
        e3_2 = self.expansion(r3_1, 3)
        conv3_2 = Conv2D(64*self.scale, 3, padding='valid', name='conv3_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e3_2)
        bn3_2 = BatchNormalization(name='bn3_2')(conv3_2)
        r3_2 = Activation('act_func1')(bn3_2)
        p3 = MaxPooling2D(pool_size = 2, strides = 2)(r3_2)

        e4_1 = self.expansion(p3, 3)
        conv4_1 = Conv2D(128*self.scale, 3, padding='valid', name='conv4_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e4_1)
        bn4_1 = BatchNormalization(name='bn4_1')(conv4_1)
        r4_1 = Activation('act_func1')(bn4_1)
        e4_2 = self.expansion(r4_1, 3)
        conv4_2 = Conv2D(128*self.scale, 3, padding='valid', name='conv4_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e4_2)
        bn4_2 = BatchNormalization(name='bn4_2')(conv4_2)
        r4_2 = Activation('act_func1')(bn4_2)
        p4 = MaxPooling2D(pool_size = 2, strides = 2)(r4_2)

        e5_1 = self.expansion(p4, 3)
        conv5_1 = Conv2D(256*self.scale, 3, padding='valid', name='conv5_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e5_1)
        bn5_1 = BatchNormalization(name='bn5_1')(conv5_1)
        r5_1 = Activation('act_func1')(bn5_1)
        e5_2 = self.expansion(r5_1, 3)
        conv5_2 = Conv2D(256*self.scale, 3, padding='valid', name='conv5_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e5_2)
        bn5_2 = BatchNormalization(name='bn5_2')(conv5_2)
        r5_2 = Activation('act_func1')(bn5_2)
        p5 = MaxPooling2D(pool_size = 2, strides = 2)(r5_2)

        e6_1 = self.expansion(p5, 3)
        conv6_1 = Conv2D(512*self.scale, 3, padding='valid', name='conv6_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e6_1)
        bn6_1 = BatchNormalization(name='bn6_1')(conv6_1)
        r6_1 = Activation('act_func1')(bn6_1)
        e6_2 = self.expansion(r6_1, 3)
        conv6_2 = Conv2D(512*self.scale, 3, padding='valid', name='conv6_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e6_2)
        bn6_2 = BatchNormalization(name='bn6_2')(conv6_2)
        r6_2 = Activation('act_func1')(bn6_2)

        tconv1 = Conv2DTranspose(256*self.scale, kernel_size=(2,2), strides=(2,2), name='tconv1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(r6_2)
        bn7_1 = BatchNormalization(name='bn7_1')(tconv1)
        rt1 = Activation('act_func1')(bn7_1)
        c1 = Concatenate(axis=3)([r5_2, rt1])
        e7_1 = self.expansion(c1, 3)
        conv7_1 = Conv2D(256*self.scale, 3, padding='valid', name='conv7_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e7_1)
        bn7_2 = BatchNormalization(name='bn7_2')(conv7_1)
        r7_1 = Activation('act_func1')(bn7_2)
        e7_2 = self.expansion(r7_1, 3)
        conv7_2 = Conv2D(256*self.scale, 3, padding='valid', name='conv7_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e7_2)
        bn7_3 = BatchNormalization(name='bn7_3')(conv7_2)
        r7_2 = Activation('act_func1')(bn7_3)

        tconv2 = Conv2DTranspose(128*self.scale, kernel_size=2, strides=2, name='tconv2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(r7_2)
        bn8_1 = BatchNormalization(name='bn8_1')(tconv2)
        rt2 = Activation('act_func1')(bn8_1)
        c2 = Concatenate(axis=3)([r4_2, rt2])
        e8_1 = self.expansion(c2, 3)
        conv8_1 = Conv2D(128*self.scale, 3, padding='valid', name='conv8_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e8_1)
        bn8_2 = BatchNormalization(name='bn8_2')(conv8_1)
        r8_1 = Activation('act_func1')(bn8_2)
        e8_2 = self.expansion(r8_1, 3)
        conv8_2 = Conv2D(128*self.scale, 3, padding='valid', name='conv8_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e8_2)
        bn8_3 = BatchNormalization(name='bn8_3')(conv8_2)
        r8_2 = Activation('act_func1')(bn8_3)

        tconv3 = Conv2DTranspose(64*self.scale, kernel_size=2, strides=2, name='tconv3', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(r8_2)
        bn9_1 = BatchNormalization(name='bn9_1')(tconv3)
        rt3 = Activation('act_func1')(bn9_1)
        c3 = Concatenate(axis=3)([r3_2, rt3])
        e9_1 = self.expansion(c3, 3)
        conv9_1 = Conv2D(64*self.scale, 3, padding='valid', name='conv9_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e9_1)
        bn9_2 = BatchNormalization(name='bn9_2')(conv9_1)
        r9_1 = Activation('act_func1')(bn9_2)
        e9_2 = self.expansion(r9_1, 3)
        conv9_2 = Conv2D(64*self.scale, 3, padding='valid', name='conv9_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e9_2)
        bn9_3 = BatchNormalization(name='bn9_3')(conv9_2)
        r9_2 = Activation('act_func1')(bn9_3)

        tconv4 = Conv2DTranspose(32*self.scale, kernel_size=2, strides=2, name='tconv4', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(r9_2)
        bn10_1 = BatchNormalization(name='bn10_1')(tconv4)
        rt4 = Activation('act_func1')(bn10_1)
        c4 = Concatenate(axis=3)([r2_2, rt4])
        e10_1 = self.expansion(c4, 3)
        conv10_1 = Conv2D(32*self.scale, 3, padding='valid', name='conv10_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e10_1)
        bn10_2 = BatchNormalization(name='bn10_2')(conv10_1)
        r10_1 = Activation('act_func1')(bn10_2)
        e10_2 = self.expansion(r10_1, 3)
        conv10_2 = Conv2D(32*self.scale, 3, padding='valid', name='conv10_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e10_2)
        bn10_3 = BatchNormalization(name='bn10_3')(conv10_2)
        r10_2 = Activation('act_func1')(bn10_3)

        tconv5 = Conv2DTranspose(16*self.scale, kernel_size=2, strides=2, name='tconv5', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(r10_2)
        bn11_1 = BatchNormalization(name='bn11_1')(tconv5)
        rt5 = Activation('act_func1')(bn11_1)
        c5 = Concatenate(axis=3)([r1_2, rt5])
        e11_1 = self.expansion(c5, 3)
        conv11_1 = Conv2D(16*self.scale, 3, padding='valid', name='conv11_1', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e11_1)
        bn11_2 = BatchNormalization(name='bn11_2')(conv11_1)
        r11_1 = Activation('act_func1')(bn11_2)
        e11_2 = self.expansion(r11_1, 3)
        conv11_2 = Conv2D(16*self.scale, 3, padding='valid', name='conv11_2', kernel_initializer=self.initializer1, kernel_regularizer=keras.regularizers.l2(self.lamda))(e11_2)
        bn11_3 = BatchNormalization(name='bn11_3')(conv11_2)
        r11_2 = Activation('act_func1')(bn11_3)

        conv12_1 = Conv2D(1, 1, padding='same', name='conv12_1', kernel_initializer=self.initializer2, kernel_regularizer=keras.regularizers.l2(self.lamda))(r11_2)
        bn12_1 = BatchNormalization(name='bn12_1')(conv12_1)
        r12_1 = Activation('linear')(bn12_1)

        return tf.keras.Model(inputs=input, outputs=r12_1)

    @tf.function
    def call(self, x):
        y = self.unet(x)

        return y

    @tf.function
    def loss(self, pred, true):
        # MSE
        p = pred
        t = true
        p = tf.reshape(p, [-1, 128, 128])
        t = tf.reshape(t, [-1, 128, 128])
        
        train_loss = tf.reduce_mean(tf.square(p-t))


        return(train_loss)
    
    @tf.function
    def loss_r(self, pred, true):
        # MRE
        p = pred
        t = true
        p = tf.reshape(p, [-1, 128, 128])
        t = tf.reshape(t, [-1, 128, 128])
        train_loss = tf.reduce_mean(tf.abs((p-t)/t))

        return(train_loss)
    
    @tf.function
    def loss_a(self, pred, true):
        # MAE
        p = pred
        t = true
        p = tf.reshape(p, [-1, 128, 128])
        t = tf.reshape(t, [-1, 128, 128])
        train_loss = tf.reduce_mean(tf.abs(p-t))

        return(train_loss)
    
    @tf.function
    def loss_s(self, pred, true):
        # MSE
        p = pred
        t = true
        p = tf.reshape(p, [-1, 128, 128])
        t = tf.reshape(t, [-1, 128, 128])
        #Total Loss
        train_loss = tf.reduce_mean(tf.square(p-t))

        return(train_loss)