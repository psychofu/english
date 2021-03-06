#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: nl8590687
用于测试整个一套语音识别系统的程序
语音模型 + 语言模型
"""
import platform as plat

from asrModel import ModelSpeech
from keras import backend as K

datapath = ''
modelpath = 'checkpoint'

system_type = plat.system() # 由于不同的系统的文件路径表示不一样，需要进行判断
if(system_type == 'Windows'):
	datapath = '.'
	modelpath = modelpath + '\\'
elif(system_type == 'Linux'):
	datapath = '.'
	modelpath = modelpath + '/'
else:
	print('*[Message] Unknown System\n')
	datapath = 'dataset'
	modelpath = modelpath + '/'

f = open('dict.txt', 'r', encoding='utf-8')
outputSize = len(f.readlines()) + 1
ms = ModelSpeech(outputSize)

ms.LoadModel(modelpath + 'weights_model.hdf5')

#ms.TestModel(datapath, str_dataset='test', data_count = 64, out_report = True)
r = ms.RecognizeSpeech_FromFile('E:\\english\\data\\avi\\MU7554\\MU7554_14.wav')


K.clear_session()

print('*[提示] 语音识别结果：\n',r)