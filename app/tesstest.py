import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import pytesseract
import shutil
import os
import random
try:
 from PIL import Image
except ImportError:
 import Image

configuration = ("-l eng --oem 1 --psm 7")

price = pytesseract.image_to_string(Image.open(r'/home/user/src/darknet/proj/eb/textboxes/dates/tb_0_12_date.jpg'), config=configuration)

print(price)