from fastapi import FastAPI
import cv2
import numpy as np
app = FastAPI()
import base64
from code import gethog


def readb64(encoded_data):
    data = encoded_data.split(',',1)
    img_str = data[1]
    decode = base64.b64decode(img_str)
    img = cv2.imdecode(np.frombuffer(decode, np.uint8),cv2.IMREAD_GRAYSCALE)
    return img




@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/api/gethog")
def read_str(item_str):
   
    img = readb64(item_str)
    hog = gethog(img)
    return {"message": hog}

