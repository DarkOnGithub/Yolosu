import websocket
import json

def on_message(ws, message):
    try:
        data = json.loads(message)
        gameplay = data.get("gameplay", {})

        combo = gameplay.get("combo", 0)
        score = gameplay.get("score", 0)
        accuracy = gameplay.get("accuracy", 0.0)

        print(f"Combo: {combo}, Score: {score}, Accuracy: {accuracy:.2f}%")
    except json.JSONDecodeError:
        print("Received non-JSON data")

def on_error(ws, error):
    print(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def on_open(ws):
    print("WebSocket connection established")

if __name__ == "__main__":
    ws = websocket.WebSocketApp("ws://127.0.0.1:24050/ws",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.run_forever()
