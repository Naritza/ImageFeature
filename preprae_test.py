import numpy as ps
import cv2
import base64
import requests
import os
import pickle

url = "http://localhost:8080:5000/api/gethog"
def img2vec(img):
    resized_img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
    v, buffer = cv2.imencode(".jpg", resized_img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    image_data_string = "data:image/jpeg;base64," + img_str
    params = {"item_str": image_data_string}
    response = requests.get(url, params=params)
    return response.json()


img_list = []


path = '..\\Cars Dataset\\test'

carvectors = []
for sub in os.listdir(path):
    for fn in os.listdir(os.path.join(path,sub)):
        img_file_name = os.path.join(path,sub)+"/"+fn
        img = cv2.imread(img_file_name)
        res = img2vec(img)
        vec = list(res["message"])
        vec.append(sub)
        carvectors.append(vec)

write_path = "carvectors_test.pkl"
pickle.dump(carvectors, open(write_path,"wb"))
print("data preparation is done")