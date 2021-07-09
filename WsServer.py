from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import queue
import numpy as np
import threading
from asrModel import ModelSpeech, GetSymbolList
a = ModelSpeech()
a.LoadModel("checkpoint/weights_1624889133.0653555.hdf5")

list_symbol_dic = GetSymbolList()  # 获取拼音列表

class WSServerInstance(WebSocket):
    def handleMessage(self):
        # 这里的server就是绑定的WSServer里的server
        self.server.data_queue.put([self.data, self.client.fileno()])

    def handleConnected(self):
        print(self.address, "connected")

    def handleClose(self):
        print(self.address, "closed")


class WSServer(object):
    def __init__(self, port):
        self.server = SimpleWebSocketServer('', port, WSServerInstance)
        self.server.data_queue = queue.Queue(0)
        self.server_thread = None
        self.run()

    def run(self):
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.start()

    def run_server(self):
        self.server.serveforever()

    def broadcast_message(self, message):
        """
            广播消息，向所有连接中的client发消息
        """
        for key, client in self.server.connections.items():
            client.sendMessage(message)

    def send_message(self, message, fileno):
        """
        向一个用户发送信息
        :param message:
        :return:
        """
        if fileno in self.server.connections.keys():
            self.server.connections[fileno].sendMessage(message)
            print("send message to: %d"%fileno)
        else:
            print("send failed, maybe connection lost")

    def main_proccess(self):
        """
           主循环，可以加一些其他流程的代码
        """
        my_buf = []
        while True:
            if (not self.server.data_queue.empty()):
                message, fileno = self.server.data_queue.get()
                if False:        # test
                    self.send_message(message, fileno)
                my_buf.append(message)
                if len(my_buf) == 10:
                    wave_data = np.fromstring(b"".join(my_buf), dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
                    wave_data.shape = -1, 1  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
                    wave_data = wave_data.T  # 将矩阵转置
                    pinyin = a.RecognizeSpeech(wave_data, 16000)
                    r_str = []
                    for i in pinyin:
                        r_str.append(list_symbol_dic[i])

                    print("\ntest phones:")
                    print(" ".join(r_str))
                    # 拼音转汉字
                    result = ""
                    if r_str:
                        result = ''.join(viterbi(hmm_params=hmmparams, observations=tuple(r_str), path_num=1)[0].path)
                        print("on get message:", result)


                    # self.broadcast_message(result)
                    self.send_message(result, fileno)
                    saveMessage = my_buf[-1]
                    my_buf.clear()
                    my_buf.append(saveMessage)


ws_server = WSServer(8888)
ws_server.main_proccess()