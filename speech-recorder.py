import wave
from pyaudio import PyAudio, paInt16
import threading

framerate = 16000  # 16000帧/s
NUM_SAMPLES = 8000  # 2000帧
channels = 1
sampwidth = 2

def my_record():
    pa = PyAudio()
    stream = pa.open(format=paInt16, channels=1,
                     rate=framerate, input=True,
                     frames_per_buffer=NUM_SAMPLES)
    my_buf = list()
    while len(my_buf) < 5:
    # while True:
        # 每8000帧传输一次,0.5s
        string_audio_data = stream.read(NUM_SAMPLES)
        my_buf.append(string_audio_data)

    save_wave_file("test.wav", my_buf)
    stream.close()
    pa.terminate()

# python播放音频文件
def play():
    wf = wave.open("data\\avi\\MU2060\\MU2060_1.wav", 'rb')
    p = PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    while True:
        data = wf.readframes(2000)
        # print(data)
        if data == "": break
        stream.write(data)
    wf.close()
    stream.close()
    p.terminate()

# 保存录音文件
def save_wave_file(filename, data):
    '''save the date to the wavfile'''
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sampwidth)
    wf.setframerate(framerate)
    wf.writeframes(b"".join(data))
    wf.close()

import requests
from general_function.file_wav import *
def http_client():
    url = 'http://121.5.66.137:20000/'
    token = 'qwertasd'
    wavsignal, fs = read_wav_data('test.wav')
    datas = {'token': token, 'fs': fs, 'wavs': wavsignal}
    r = requests.post(url, datas)
    r.encoding = 'utf-8'
    print(r.text)

if __name__ == '__main__':
    my_record()
    # http_client()
    # print('Over!')
    # play()
