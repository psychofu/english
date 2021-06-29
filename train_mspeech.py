#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于训练语音识别系统语音模型的程序

"""
import platform as plat
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


from asrModel import ModelSpeech

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#进行配置，使用95%的GPU
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


datapath = ''
modelpath = 'model_speech'


if(not os.path.exists(modelpath)): # 判断保存模型的目录是否存在
	os.makedirs(modelpath) # 如果不存在，就新建一个，避免之后保存模型的时候炸掉

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = 'datalist'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = 'datalist'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'datalist'
	modelpath = modelpath + '/'

ms = ModelSpeech()

#ms.LoadModel(modelpath + 'speech_model251_e_0_step_327500.model')

# datapath, batch_size=32, save_step=1000, epochs=20
ms.TrainModel(datapath, batch_size = 1, save_step = 1000, epochs=20)


