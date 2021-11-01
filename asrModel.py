# -*- coding: utf-8 -*-

from general_function.file_wav import *
from general_function.gen_func import *
import os, time
# LSTM_CNN
import numpy as np
import random
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Reshape, GRU
from keras.layers import Lambda, Activation, Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

from readdata24 import DataSpeech

ModelName = '251'


class ModelSpeech:  # 语音模型类
    def __init__(self, outputSize):
        '''
        初始化outputSize
        '''
        self.MS_OUTPUT_SIZE = outputSize  # 神经网络最终输出的每一个字符向量维度的大小
        self.label_max_string_length = 64
        self.AUDIO_LENGTH = 1600
        self.AUDIO_FEATURE_LENGTH = 200
        self._model, self.base_model = self.CreateModel()

    def CreateModel(self):
        '''
        定义CNN/LSTM/CTC模型，使用函数式模型
        输入层：200维的特征值序列，一条语音数据的最大长度设为1600（大约16s）
        隐藏层：卷积池化层，卷积核大小为3x3，池化窗口大小为2
        隐藏层：全连接层
        输出层：全连接层，神经元数量为self.MS_OUTPUT_SIZE，使用softmax作为激活函数，
        CTC层：使用CTC的loss作为损失函数，实现连接性时序多输出

        '''

        input_data = Input(name='audio_info', shape=(self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1))

        layer_h1 = Conv2D(32, (3, 3), use_bias=False, activation='relu', padding='same',
                          kernel_initializer='he_normal')(input_data)  # 卷积层
        layer_h1 = Dropout(0.05)(layer_h1)
        layer_h2 = Conv2D(32, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h1)  # 卷积层
        layer_h3 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h2)  # 池化层
        layer_h3 = Dropout(0.05)(layer_h3)

        layer_h4 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h3)  # 卷积层
        layer_h4 = Dropout(0.1)(layer_h4)
        layer_h5 = Conv2D(64, (3, 3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(
            layer_h4)  # 卷积层
        layer_h6 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h5)  # 池化层
        layer_h6 = Dropout(0.1)(layer_h6)

        layer_h7 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h6)  # 卷积层
        layer_h7 = Dropout(0.15)(layer_h7)
        layer_h8 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                          kernel_initializer='he_normal')(layer_h7)  # 卷积层
        layer_h9 = MaxPooling2D(pool_size=2, strides=None, padding="valid")(layer_h8)  # 池化层
        layer_h9 = Dropout(0.15)(layer_h9)

        layer_h10 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h9)  # 卷积层
        layer_h10 = Dropout(0.2)(layer_h10)
        layer_h11 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h10)  # 卷积层
        layer_h12 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h11)  # 池化层
        layer_h12 = Dropout(0.2)(layer_h12)

        layer_h13 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h12)  # 卷积层
        layer_h13 = Dropout(0.2)(layer_h13)
        layer_h14 = Conv2D(128, (3, 3), use_bias=True, activation='relu', padding='same',
                           kernel_initializer='he_normal')(layer_h13)  # 卷积层
        layer_h15 = MaxPooling2D(pool_size=1, strides=None, padding="valid")(layer_h14)  # 池化层

        layer_h16 = Reshape((200, 3200))(layer_h15)  # Reshape层
        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        model_data = Model(inputs=input_data, outputs=y_pred)
        model_data.summary()

        labels = Input(name='the_labels', shape=[self.label_max_string_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pred, labels, input_length, label_length])
        model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        model.summary()

        # 输出模型png图
        # keras.utils.plot_model(model, 'model.png', show_shapes=True)
        # keras.utils.plot_model(model_data, 'model_data.png', show_shapes=True)
        # decay：浮动> = 0.每次更新时学习率下降。
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0001, epsilon=10e-8)
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)

        # print('[*提示] 创建模型成功，模型编译成功')
        print('[*Info] Create Model Successful, Compiles Model Successful. ')
        return model, model_data

    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, :, :]
        # length:  y_pred > input_length > label_length
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def TrainModel(self, datapath, batch_size=32, save_step=1000, epochs=20):
        data = DataSpeech(datapath, 'train')
        yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)
        checkpointDIR = "checkpoint"
        if not os.path.exists(checkpointDIR):
            os.makedirs(checkpointDIR)
        checkpoint = ModelCheckpoint("checkpoint/weights_model" + ".hdf5", monitor="loss", verbose=0,
                                     save_best_only=True)
        tensorboard = TensorBoard(batch_size=batch_size, update_freq="batch")

        try:
            # self._model.fit_generator(yielddatas, save_step, nb_worker=2)  考虑：train_on_batch
            history = self._model.fit_generator(yielddatas, steps_per_epoch=save_step, epochs=epochs,
                                                callbacks=[checkpoint, tensorboard])
            # print(history.history())
        except StopIteration:
            print('[error] generator error. please check data format.')
        # 保存weights
        self.SaveModel('savaModel')

    def LoadModel(self, filename):
        '''
        加载模型参数
        '''
        self._model.load_weights(filename)
        # self.base_model.load_weights(filename + '.base')
        print("load model success")

    def SaveModel(self, filename):
        '''
        保存模型参数
        '''
        self._model.save_weights(filename + '.model')
        self.base_model.save_weights(filename + '.model.base')

    def Predict(self, data_input, input_len):
        '''
        预测结果
        返回语音识别后的拼音符号列表
        '''

        batch_size = 1
        in_len = np.zeros((batch_size), dtype=np.int32)

        in_len[0] = input_len

        x_in = np.zeros((batch_size, self.AUDIO_LENGTH, self.AUDIO_FEATURE_LENGTH, 1), dtype=np.float)

        for i in range(batch_size):
            x_in[i, 0:len(data_input)] = data_input

        base_pred = self.base_model.predict(x=x_in)

        base_pred = base_pred[:, :, :]

        # 防止多次调用ctc-decode 的get_value 导致内存持续增大 --------------------------------------------
        ctc_class = Ctc_Decode(batch_size=batch_size, timestep=200, nclass=829)
        predict_y = ctc_class.ctc_decode_tf([base_pred, np.reshape(input_len, (1, 1))])  # ctc解码
        r1 = predict_y[0]
        # ------------------------------------------------------------------------------------------
        # r = K.ctc_decode(base_pred, in_len, greedy=True, beam_width=100, top_paths=1)
        #
        # r1 = K.get_value(r[0][0])
        #
        # r1 = r1[0]

        return r1

    def RecognizeSpeech(self, wavsignal, fs):
        '''
        最终做语音识别用的函数，识别一个wav序列的语音
        '''

        data_input = GetFrequencyFeature3(wavsignal, fs)

        input_length = len(data_input)
        input_length = input_length // 8 + input_length % 8
        data_input = np.array(data_input, dtype=np.float)
        data_input = data_input.reshape(data_input.shape[0], data_input.shape[1], 1)
        # if len(data_input) > 1600:
        #     print("data_input length is : %d, wrong" % len(data_input))
        #     return None
        r1 = self.Predict(data_input, input_length)
        print(r1)
        list_symbol_dic = GetSymbolList()  # 获取拼音列表

        r_str = []
        for i in r1:
            r_str.append(list_symbol_dic[i])

        return r_str
        # return r1

    def RecognizeSpeech_FromFile(self, filename):
        '''
        最终做语音识别用的函数，识别指定文件名的语音
        '''

        wavsignal, fs = read_wav_data(filename)

        r = self.RecognizeSpeech(wavsignal, fs)

        return r

        pass

    @property
    def model(self):
        '''
        返回keras model
        '''
        return self._model


class StopTraining(keras.callbacks.Callback):
    def __init__(self, thres):
        super(StopTraining, self).__init__()
        self.thres = thres

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        if logs.get('loss') < self.thres:
            self.model.stop_training = True


class Ctc_Decode:
    # 用tf定义一个专门ctc解码的图和会话，就不会一直增加节点了，速度快了很多
    def __init__(self, batch_size, timestep, nclass):
        self.batch_size = batch_size
        self.timestep = timestep
        self.nclass = nclass
        self.graph_ctc = tf.Graph()
        with self.graph_ctc.as_default():
            self.y_pred_tensor = tf.placeholder(shape=[self.batch_size, self.timestep, self.nclass], dtype=tf.float32,
                                                name="y_pred_tensor")
            self._y_pred_tensor = tf.transpose(self.y_pred_tensor, perm=[1, 0, 2])  # 要把timestep 放在第一维
            self.input_length_tensor = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32,
                                                      name="input_length_tensor")
            self._input_length_tensor = tf.squeeze(self.input_length_tensor, axis=1)  # 传进来的是 [batch_size,1] 所以要去掉一维
            self.ctc_decode, _ = tf.nn.ctc_greedy_decoder(self._y_pred_tensor, self._input_length_tensor)
            self.decoded_sequences = tf.sparse_tensor_to_dense(self.ctc_decode[0])

            self.ctc_sess = tf.Session(graph=self.graph_ctc)

    def ctc_decode_tf(self, args):
        y_pred, input_length = args
        decoded_sequences = self.ctc_sess.run(self.decoded_sequences,
                                              feed_dict={self.y_pred_tensor: y_pred,
                                                         self.input_length_tensor: input_length})
        return decoded_sequences


import os


def test():
    a = ModelSpeech()
    a.LoadModel("checkpoint/weights_1596110931.0748937.hdf5")

    # 用字典存储test内容
    test_cont = dict()
    f = open("dataset/ibingli/test.syllable.txt", mode="r", encoding="utf-8")
    fcont = f.readlines()
    f.close()
    for l in fcont:
        l.strip("\n")
        l = l.split("\t")
        test_cont[l[0].split(".")[0]] = l[1]

    testdir = os.listdir("test_wav")
    testdir = [i.split(".")[0] for i in testdir]
    for wavfile in testdir:
        pinyin = a.RecognizeSpeech_FromFile("test_wav/" + wavfile + ".wav")
        list_symbol_dic = GetSymbolList()  # 获取拼音列表
        r_str = []
        for i in pinyin:
            r_str.append(list_symbol_dic[i])

        print("\ntest phones:")
        print(" ".join(r_str))
        print("\ntrue phones:")
        print(test_cont[wavfile])


def GetSymbolList():
    with open('dict.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.replace('\n', "") for line in lines]
        return lines

if __name__ == "__main__":
    a = ModelSpeech(829)
    a.LoadModel("checkpoint/weights_model.hdf5")

    pinyin = a.RecognizeSpeech_FromFile("data/avi//MU204/MU204_14.wav")

    print("\ntest phones:")
    print(pinyin)
