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
    def __init__(self):
        '''
        初始化
        默认输出的拼音的表示大小是1424，即1423个拼音+1个空白块
        '''
        self.MS_OUTPUT_SIZE = 682  # 神经网络最终输出的每一个字符向量维度的大小
        # self.label_max_string_length = 128
        self.label_max_string_length = 64
        # self.AUDIO_LENGTH = 5600
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
        # layer_h16 = Reshape((700, 3200))(layer_h15)  # Reshape层
        # layer_h16 = GRU(256, activation='relu', use_bias=True, kernel_initializer='he_normal')(layer_h16)
        # layer_h5 = LSTM(256, activation='relu', use_bias=True, return_sequences=True)(layer_h4) # LSTM层
        # layer_h6 = Dropout(0.2)(layer_h5) # 随机中断部分神经网络连接，防止过拟合
        layer_h16 = Dropout(0.3)(layer_h16)
        layer_h17 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h16)  # 全连接层
        layer_h17 = Dropout(0.3)(layer_h17)
        layer_h18 = Dense(self.MS_OUTPUT_SIZE, use_bias=True, kernel_initializer='he_normal')(layer_h17)  # 全连接层

        y_pred = Activation('softmax', name='Activation0')(layer_h18)
        model_data = Model(inputs=input_data, outputs=y_pred)
        # model_data.summary()

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
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def TrainModel(self, datapath, batch_size=32, save_step=1000, epochs=20):
        """
        训练模型
        参数：
            datapath: 数据保存的路径
            epoch: 迭代轮数
            save_step: 每多少步保存一次模型
            filename: 默认保存文件名，不含文件后缀名
        """
        data = DataSpeech(datapath, 'train')
        yielddatas = data.data_genetator(batch_size, self.AUDIO_LENGTH)
        checkpointDIR = "checkpoint"
        if not os.path.exists(checkpointDIR):
            os.makedirs(checkpointDIR)
        checkpoint = ModelCheckpoint("checkpoint/weights_" + str(time.time()) + ".hdf5", monitor="loss", verbose=0,
                                     save_best_only=True)
        tensorboard = TensorBoard(batch_size=batch_size, update_freq="batch")

        try:
            # print('[message] epoch %d . Have train datas %d+' % (epoch, n_step * save_step))
            # data_genetator是一个生成器函数

            # self._model.fit_generator(yielddatas, save_step, nb_worker=2)  考虑：train_on_batch
            history = self._model.fit_generator(yielddatas, steps_per_epoch=save_step, epochs=epochs,
                                                callbacks=[checkpoint, tensorboard])
            # n_step += 1
        except StopIteration:
            print('[error] generator error. please check data format.')
            # break
        # self.SaveModel(filename='model_speech/speech_model251_e_' + str(epoch))
        self.TestModel(datapath, str_dataset='test', data_count=4)
        # average = sum(history.history["loss"]) / len(history.history["loss"])
        # while min(history.history["loss"]) > 50:
        #     history = self._model.fit_generator(yielddatas, steps_per_epoch=save_step, epochs=epochs,
        #                                              callbacks=[checkpoint, tensorboard])
        #     self.TestModel(datapath, str_dataset='test', data_count=4)
        # average = sum(history.history["loss"]) / len(history.history["loss"])

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
        # 需要安装 hdf5 模块
        # self._model.save(filename + '.h5')
        # self.base_model.save(filename + '.base.h5')
        f = open('step' + ModelName + '.txt', 'w')
        f.write(filename)
        f.close()

    def TestModel(self, datapath, str_dataset='dev', data_count=32, out_report=False, show_ratio=True,
                  io_step_print=10, io_step_file=10):
        '''
        测试检验模型效果

        io_step_print
            为了减少测试时标准输出的io开销，可以通过调整这个参数来实现

        io_step_file
            为了减少测试时文件读写的io开销，可以通过调整这个参数来实现

        '''
        data = DataSpeech(datapath, str_dataset)
        num_data = data.GetDataNum()  # 获取数据的数量
        if data_count <= 0 or data_count > num_data:  # 当data_count为小于等于0或者大于测试数据量的值时，则使用全部数据来测试
            data_count = num_data

        try:
            ran_num = random.randint(0, num_data - 1)  # 获取一个随机数

            words_num = 0
            word_error_num = 0

            nowtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            if out_report:
                txt_obj = open('Test_Report_' + str_dataset + '_' + nowtime + '.txt', 'w', encoding='UTF-8')  # 打开文件并读入

            txt = '测试报告\n模型编号 ' + ModelName + '\n\n'
            for i in range(data_count):
                data_input, data_labels = data.GetData((ran_num + i) % num_data)  # 从随机数开始连续向后取一定数量数据

                # 数据格式出错处理 开始
                # 当输入的wav文件长度过长时自动跳过该文件，转而使用下一个wav文件来运行
                num_bias = 0
                while data_input.shape[0] > self.AUDIO_LENGTH:
                    print('*[Error]', 'wave data lenghth of num', (ran_num + i) % num_data, 'is too long.',
                          '\n A Exception raise when test Speech Model.')
                    num_bias += 1
                    data_input, data_labels = data.GetData((ran_num + i + num_bias) % num_data)  # 从随机数开始连续向后取一定数量数据
                # 数据格式出错处理 结束

                pre = self.Predict(data_input, data_input.shape[0] // 8)

                words_n = data_labels.shape[0]  # 获取每个句子的字数
                words_num += words_n  # 把句子的总字数加上
                edit_distance = GetEditDistance(data_labels, pre)  # 获取编辑距离
                if edit_distance <= words_n:  # 当编辑距离小于等于句子字数时
                    word_error_num += edit_distance  # 使用编辑距离作为错误字数
                else:  # 否则肯定是增加了一堆乱七八糟的奇奇怪怪的字
                    word_error_num += words_n  # 就直接加句子本来的总字数就好了

                if (i % io_step_print == 0 or i == data_count - 1) and show_ratio == True:
                    # print('测试进度：',i,'/',data_count)
                    print('Test Count: ', i, '/', data_count)

                if out_report == True:
                    if i % io_step_file == 0 or i == data_count - 1:
                        txt_obj.write(txt)
                        txt = ''

                    txt += str(i) + '\n'
                    txt += 'True:\t' + str(data_labels) + '\n'
                    txt += 'Pred:\t' + str(pre) + '\n'
                    txt += '\n'

            # print('*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率：', word_error_num / words_num * 100, '%')
            print('*[Test Result] Speech Recognition ' + str_dataset + ' set word error ratio: ',
                  word_error_num / words_num * 100, '%')
            if out_report:
                txt += '*[测试结果] 语音识别 ' + str_dataset + ' 集语音单字错误率： ' + str(word_error_num / words_num * 100) + ' %'
                txt_obj.write(txt)
                txt = ''
                txt_obj.close()

        except StopIteration:
            print('[Error] Model Test Error. please check data format.')

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
        ctc_class = Ctc_Decode(batch_size=batch_size, timestep=200, nclass=682)
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


# from Pinyin2Hanzi import DefaultHmmParams
# from Pinyin2Hanzi import viterbi
#
# hmmparams = DefaultHmmParams()
# if __name__ == "__main__":
#     a = ModelSpeech()
#     a.LoadModel("model_speech/speech_model251_e_0_step_1361250.model")
#
#     pinyin = a.RecognizeSpeech_FromFile("test.wav")
#     list_symbol_dic = GetSymbolList()  # 获取拼音列表
#     r_str = []
#     for i in pinyin:
#         r_str.append(list_symbol_dic[i])
#
#     print("\ntest phones:")
#     print(" ".join(r_str))
#
#     # 拼音转汉字
#     result = ''.join(viterbi(hmm_params=hmmparams, observations=tuple(r_str), path_num=1)[0].path)
#     print(result)
