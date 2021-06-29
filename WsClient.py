# coding:utf-8

import websocket
import json
import time
import threading


class WebsocketClient(object):
    """docstring for WebsocketClient"""

    def __init__(self, address, message_callback=None):
        super(WebsocketClient, self).__init__()
        self.address = address
        self.message_callback = message_callback

    def on_message(self, message):
        # message = json.loads(message)
        print("on_client_message:", message)
        if self.message_callback:
            self.message_callback(message)

    def on_error(self, error):
        print("client error:", error)

    def on_close(self):
        print("### client closed ###")
        self.ws.close()
        self.is_running = False

    def on_open(self):
        self.is_running = True
        print("on open")

    def close_connect(self):
        self.ws.close()

    def send_message(self, message):
        try:
            self.ws.send(message)
        except BaseException as err:
            pass

    def run(self):
        websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(self.address,
                                         on_message=lambda ws, message: self.on_message(message),
                                         on_error=lambda ws, error: self.on_error(error),
                                         on_close=lambda ws: self.on_close())
        self.ws.on_open = lambda ws: self.on_open(ws)
        self.is_running = False
        while True:
            print(self.is_running)
            if not self.is_running:
                self.ws.run_forever()
            time.sleep(3)


class WSClient(object):
    def __init__(self, address, call_back):
        super(WSClient, self).__init__()
        self.client = WebsocketClient(address, call_back)
        self.client_thread = None

    def run(self):
        self.client_thread = threading.Thread(target=self.run_client)
        self.client_thread.start()

    def run_client(self):
        self.client.run()

    def send_message(self, message):
        self.client.send_message(message)


ws_client = WSClient("ws://localhost:8888", lambda message: print("call_back message:", message))
ws_client.run()

while True:
    ws_client.send_message("time:" + str(time.time()))
    time.sleep(1)