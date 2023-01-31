from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout, Reshape
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Model
import numpy as np

class DeepAnt(Model):

    def __init__(self, 
                 w_l: int = 100, 
                 f_h: int = 1, 
                 n_features:int = 1, 
                 kernel_size: int = 3, 
                 n_filter_1: int = 32, 
                 n_filter_2: int = 32, 
                 neurons_dense:int = 40, 
                 conv_stride: int = 1, 
                 pool_size_1: int = 2,
                 pool_size_2: int = 2, 
                 pool_stride_1: int = 2, 
                 pool_stride_2: int = 2, 
                 dropout_rate: float = 0.25
                 ):
        
        super().__init__()
        
        self.w_l = w_l
        self.f_h = f_h
        self.n_features = n_features
        self.kernel_size = kernel_size
        self.n_filter_1 = n_filter_1
        self.n_filter_2 = n_filter_2
        self.neurons_dense = neurons_dense
        self.conv_stride = conv_stride
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.pool_stride_1 = pool_stride_1
        self.pool_stride_2 = pool_stride_2
        self.dropout_rate = dropout_rate

        self.conv_block1 = Sequential([Conv1D(filters = self.n_filter_1, kernel_size = self.kernel_size, strides = self.conv_stride, padding = 'valid', activation = 'relu', 
                                              input_shape=(self.w_l, self.n_features)),
                                                MaxPooling1D(pool_size = self.pool_size_1)], name = 'Conv_block_1')
        self.conv_block2 = Sequential([Conv1D(filters = self.n_filter_2, kernel_size = self.kernel_size, strides = self.conv_stride, padding = 'valid', activation = 'relu'),
                                                MaxPooling1D(pool_size = self.pool_size_2)], name = 'Conv_block_2')
        self.flatten = Flatten()
        self.dense1 = Dense(units = self.neurons_dense, activation = 'relu')
        self.dropout = Dropout(self.dropout_rate)
        self.denseout = Dense(units = self.f_h)
        self.rsp = Reshape([1, -1])

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.denseout(x)
        return self.rsp(x)    

    def get_loss(self, inputs):
        yhat = self.predict(inputs)
        y = np.array([y for _, y in inputs])
        return yhat, np.linalg.norm(y.reshape(-1,1)-yhat.reshape(-1,1), axis = -1)


