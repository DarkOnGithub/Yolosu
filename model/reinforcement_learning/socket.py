import websocket
import threading
import json
import time
import subprocess
import socket

TOSU_PATH = "tosu/tosu.exe"

class Socket:
    def __init__(self, socket_url, callback):
        self.callback = callback
        self.socket_url = socket_url
        self.ws = None
        self.connected = False

    def start_tosu(self):
        subprocess.Popen([TOSU_PATH])

    def connect(self, timeout=5.0):
        self.ws = websocket.WebSocketApp(
            self.socket_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )

        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        start = time.time()
        while not self.connected:
            if time.time() - start > timeout:
                raise TimeoutError(f"WebSocket failed to open in {timeout}s")
            time.sleep(0.01)

    def _on_open(self, ws):
        print(ws.sock)
        ws.sock.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print("WebSocket connection established")
        self.connected = True
        self.start_tosu()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            self.callback(data)
        except Exception as e:
            print(f"Error processing websocket message: {e}")

    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
        self.connected = False
