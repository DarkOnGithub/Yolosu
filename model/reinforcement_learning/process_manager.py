import pyautogui
import psutil
import time

TARGET_PROCESS_NAME = "osu!.exe"

class ProcessManager:
    def __init__(self):
        self.process_target = None
        self.process_target_name = None
        self.is_bound = False

    def get_process_target(self):
        for process in psutil.process_iter(['name']):
            if "osu" in process.info['name']:
                print(process.info['name'])
            if process.info['name'] == TARGET_PROCESS_NAME:
                return process
        return None

    def bind_to_process(self):
        """Bind to the target process and ensure it's active."""
        self.process_target = self.get_process_target()
        if self.process_target:
            self.process_target_name = self.process_target.info['name']
            self.is_bound = True
            return True
        return False

    def ensure_process_active(self):
        """Ensure the process is still running and bound."""
        if not self.is_bound or not self.process_target:
            return self.bind_to_process()
        
        try:
            if not self.process_target.is_running():
                return self.bind_to_process()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return self.bind_to_process()

    def send_key(self, key):
        """Send a keyboard input to the process."""
        if self.ensure_process_active():
            pyautogui.press(key)
            return True
        return False

    def send_mouse_click(self, x=None, y=None, button='left'):
        """Send a mouse click to the process."""
        if self.ensure_process_active():
            if x is not None and y is not None:
                pyautogui.click(x=x, y=y, button=button)
            else:
                pyautogui.click(button=button)
            return True
        return False

    def send_mouse_move(self, x, y, duration=0.0):
        """Move the mouse to specified coordinates."""
        if self.ensure_process_active():
            pyautogui.moveTo(x, y, duration=duration)
            return True
        return False

    def send_key_combination(self, keys):
        """Send a combination of keys."""
        if self.ensure_process_active():
            pyautogui.hotkey(*keys)
            return True
        return False
