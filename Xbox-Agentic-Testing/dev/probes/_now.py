import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.adapters import HardwareBridge
from core.config import Config, load_dotenv_if_present
from tools.gameplay_engine import grab_nonblank
import cv2, pytesseract
from PIL import Image
load_dotenv_if_present(ROOT / ".env")
s = Config.load_all(ROOT / "config", {"settings": "settings.yaml"}, base=ROOT)["settings"]
cam = HardwareBridge(s).capture()
f, blank = grab_nonblank(cam)
print("frame", None if f is None else f.shape)
cv2.imwrite(str(ROOT/"scratch"/"live_now.png"), f)
im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
print(pytesseract.image_to_string(im.crop((0,900,1920,1080)), config="--psm 6"))
print(pytesseract.image_to_string(im.crop((300,150,1620,900)), config="--psm 6")[:600])
