#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from general_function.file_wav import *

# usl = 'http://121.5.66.137:20000/'
url = 'http://127.0.0.1:20000/'

token = 'qwertasd'

wavsignal,fs=read_wav_data('data\\MU204\\MU204_11.wav')

#print(wavsignal,fs)

datas={'token':token, 'fs':fs, 'wavs':wavsignal}

r = requests.post(url, datas)

r.encoding='utf-8'

print(r.text)