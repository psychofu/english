#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import http.server
import re

import numpy as np

from asrModel import ModelSpeech

datapath = './'
modelpath = 'checkpoint/'
ms = ModelSpeech()
ms.LoadModel(modelpath + 'weights_1624889133.0653555.hdf5')

class TestHTTPHandle(http.server.BaseHTTPRequestHandler):  
	def setup(self):
		self.request.settimeout(10)
		http.server.BaseHTTPRequestHandler.setup(self)
	
	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()
		
	def do_GET(self):  
	
		buf = 'ASRT_SpeechRecognition API'  
		self.protocal_version = 'HTTP/1.1'   
		
		self._set_response()

		buf = bytes(buf,encoding="utf-8")
		self.wfile.write(buf) 
		
	def do_POST(self):  
		'''
		处理通过POST方式传递过来并接收的语音数据
		'''
		#获取post提交的数据  
		datas = self.rfile.read(int(self.headers['content-length']))
		# 根据音频数据来进行处理
		RIFFindex = re.search(b"RIFF", datas).span()[0]
		headInfo = datas[:RIFFindex]
		bodyInfo = datas[RIFFindex:]

		if re.search(b'name="token"\r\n\r\nasr', headInfo) is not None:
			if bodyInfo.find(b"\x00LIST") != -1:
				wavData = bodyInfo[78:]
			elif bodyInfo.find(b"\x00INFOISFT"):
				wavData = bodyInfo[44:]
			# 否则数据不符合
			else:
				self._set_response()
				buf = 'Data error'
				print(buf)
				buf = bytes(buf, encoding="utf-8")
				self.wfile.write(buf)
				return
			wave_data = np.fromstring(wavData, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
			wave_data.shape = -1, 1  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
			wave_data = wave_data.T  # 将矩阵转置

			buf = self.recognize(wave_data, 16000)
			self._set_response()
			print(buf)
			buf = bytes(" ".join(buf), encoding="utf-8")
			self.wfile.write(buf)
		else:
			self._set_response()
			buf = 'Token error'
			print(buf)
			buf = bytes(buf, encoding="utf-8")
			self.wfile.write(buf)

	def recognize(self, wavs, fs):
		r=''
		try:
			r_speech = ms.RecognizeSpeech(wavs, fs)
			print(r_speech)
			str_pinyin = r_speech
		except:
			str_pinyin=''
			print('[*Message] Server raise a bug. ')
		return str_pinyin
		pass
	
	def recognize_from_file(self, filename):
		pass

import socket

class HTTPServerV6(http.server.HTTPServer):
	address_family = socket.AF_INET6

def start_server(ip, port):  
	
	if(':' in ip):
		http_server = HTTPServerV6((ip, port), TestHTTPHandle)
	else:
		http_server = http.server.HTTPServer((ip, int(port)), TestHTTPHandle)
	
	print('服务器已开启')
	
	try:
		http_server.serve_forever() #设置一直监听并接收请求  
	except KeyboardInterrupt:
		pass
	http_server.server_close()
	print('HTTP server closed')
	
if __name__ == '__main__':
	start_server('', 20000) # For IPv4 Network Only
	#start_server('::', 20000) # For IPv6 Network