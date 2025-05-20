import requests
import os
import zipfile
import io
import platform
import subprocess
PLATFORM_MAP = {
    "Windows": "win",
    "Linux": "linux",
}

API_URL = "https://api.github.com/repos/Wieku/danser-go/releases/latest"
BIN_DIR = "danser"

def install_danser():
    os_type = PLATFORM_MAP.get(platform.system())
    if not os_type:
        raise Exception("Unsupported OS")
    print("Fetching latest danser-go release info...")
    r = requests.get(API_URL)
    r.raise_for_status()
    release = r.json()
    assets = release["assets"]

    asset = next((a for a in assets if os_type in a["name"].lower() and a["name"].endswith(".zip")), None)
    if not asset:
        raise Exception(f"No suitable danser binary found for platform: {os_type}")

    url = asset["browser_download_url"]
    print(f"Downloading {asset['name']}...")

    r = requests.get(url)
    r.raise_for_status()

    print("Extracting...")
    with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
        if os.path.exists(BIN_DIR):
            for root, dirs, files in os.walk(BIN_DIR, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        os.makedirs(BIN_DIR, exist_ok=True)
        zip_ref.extractall(BIN_DIR)
    print("danser-go installed to", BIN_DIR)
    subprocess.run([os.path.join(BIN_DIR, "danser-cli.exe")])

if not os.path.exists("danser"):
    install_danser()




